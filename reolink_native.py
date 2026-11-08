"""Live 4K streams from a Reolink NVR over Reolink's native protocol.

The cameras behind the NVR send their 3840x2160 main streams as H.265. The
RLN8-410-E (hardware H3MB16, final firmware v2.0.0.280) can't relay H.265 over
RTSP, RTMP or FLV, but its native "Baichuan" protocol on the media port (9000),
the one Reolink's own apps use, does. This module implements just enough of
that protocol to log in, start a live preview of one channel and unpack the
media packets; the protocol details follow the open-source neolink and
reolink_aio projects. Frames are decoded with PyAV, on the GPU where possible.

The NVR serves at most two main streams at a time and answers a third preview
request with code 400, so HDStreamManager starts streams on demand, stops idle
ones and enforces a cap (REOLINK_HD_MAX_STREAMS in .env).

For a full-resolution still without a stream, use NvrSnapshotClient (HTTP
cmd=Snap, about 1.6 s per image on the RLN8-410-E).

Quick check from the command line (reads the NVR settings from .env):
    python reolink_native.py <channel 0-3> [seconds]
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import hashlib
import os
import re
import socket
import struct
import sys
import threading
import time
import uuid
from collections import deque, namedtuple

import av
import cv2
import numpy as np
import requests

# --- Protocol constants -------------------------------------------------------
MAGIC = 0x0ABCDEF0
MAGIC_REV = 0x0FEDCBA0
# magic, message id, body length, channel, stream type, message number, response code, class
_HEADER = struct.Struct('<IIIBBHHH')
_CLASSES_WITH_PAYLOAD_OFFSET = (0x6414, 0x0000)
MSG_LOGIN, MSG_VIDEO, MSG_VIDEO_STOP = 1, 3, 4
# Reply code of the legacy login -> encryption the device picked
_ENCRYPTION = {0xDD00: 'none', 0xDD01: 'bc', 0xDD02: 'aes', 0xDD12: 'fullaes'}
_XOR_KEY = bytes([0x1F, 0x2D, 0x3C, 0x4B, 0x5A, 0x69, 0x78, 0xFF])
_AES_IV = b'0123456789abcdef'
STREAM_HANDLES = {'mainStream': 0, 'subStream': 256, 'externStream': 1024}
_MAX_BODY = 16 * 1024 * 1024

_LOGIN_XML = ('<?xml version="1.0" encoding="UTF-8" ?>\n<body>\n<LoginUser version="1.1">\n'
              '<userName>{user}</userName>\n<password>{password}</password>\n<userVer>1</userVer>\n'
              '</LoginUser>\n<LoginNet version="1.1">\n<type>LAN</type>\n<udpPort>0</udpPort>\n'
              '</LoginNet>\n</body>\n')
_PREVIEW_XML = ('<?xml version="1.0" encoding="UTF-8" ?>\n<body>\n<Preview version="1.1">\n'
                '<channelId>{channel}</channelId>\n<handle>{handle}</handle>\n'
                '<streamType>{stream}</streamType>\n</Preview>\n</body>\n')

BcMessage = namedtuple('BcMessage', 'msg_id channel num resp cls payload_offset body')


class ReolinkError(Exception):
    """Talking to the NVR failed."""


class LoginFailed(ReolinkError):
    """The NVR rejected the login."""


class StreamRefused(ReolinkError):
    """The NVR answered the preview request with an error code."""

    def __init__(self, code):
        super().__init__(f'the NVR refused the stream (code {code})')
        self.code = code


class HDLimitReached(ReolinkError):
    """Starting another 4K stream would exceed the NVR's limit."""

    def __init__(self, message, active):
        super().__init__(message)
        self.active = active


class HDUnavailable(ReolinkError):
    """A 4K stream could not be started."""


# --- Encryption ---------------------------------------------------------------
def _bc_xor(offset, data):
    """Reolink's XOR "BC" encoding, keyed by the header's channel byte."""
    if not data:
        return b''
    pattern = np.frombuffer(bytes(_XOR_KEY[(offset + i) % 8] ^ (offset & 0xFF) for i in range(8)), np.uint8)
    arr = np.frombuffer(bytes(data), np.uint8)
    return (arr ^ np.resize(pattern, arr.size)).tobytes()


def _load_libcrypto():
    """OpenSSL's libcrypto: Python ships it on Windows, Linux/macOS have a system copy."""
    import ssl  # noqa: F401  (loads Python's bundled OpenSSL into the process)
    names = []
    if sys.platform == 'win32':
        for base in dict.fromkeys((sys.base_prefix, sys.prefix, os.path.dirname(sys.executable))):
            names += [os.path.join(base, 'DLLs', dll) for dll in
                      ('libcrypto-3-x64.dll', 'libcrypto-3.dll', 'libcrypto-1_1-x64.dll', 'libcrypto-1_1.dll')]
    found = ctypes.util.find_library('crypto') or ctypes.util.find_library('libcrypto')
    if found:
        names.append(found)
    names += ['libcrypto.so.3', 'libcrypto.so.1.1', 'libcrypto.3.dylib']
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            lib.EVP_CIPHER_CTX_new.restype = ctypes.c_void_p
            lib.EVP_aes_128_cfb128.restype = ctypes.c_void_p
            lib.EVP_CipherInit_ex.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
            lib.EVP_CipherUpdate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                             ctypes.c_char_p, ctypes.c_int]
            lib.EVP_CIPHER_CTX_free.argtypes = [ctypes.c_void_p]
            return lib
        except (OSError, AttributeError):
            continue
    return None


@functools.lru_cache(maxsize=1)
def _aes_cfb():
    """AES-128-CFB as fn(key, data, encrypt) -> bytes, from `cryptography` or OpenSSL."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        def aes(key, data, encrypt):
            cipher = Cipher(algorithms.AES(key), modes.CFB(_AES_IV))
            ctx = cipher.encryptor() if encrypt else cipher.decryptor()
            return ctx.update(data) + ctx.finalize()
        return aes
    except ImportError:
        pass
    lib = _load_libcrypto()
    if lib is None:
        return None

    def aes(key, data, encrypt):
        ctx = lib.EVP_CIPHER_CTX_new()
        if not ctx:
            raise MemoryError('EVP_CIPHER_CTX_new failed')
        try:
            out = ctypes.create_string_buffer(len(data) + 16)
            n = ctypes.c_int(0)
            if (lib.EVP_CipherInit_ex(ctx, lib.EVP_aes_128_cfb128(), None, key, _AES_IV, int(encrypt)) != 1
                    or lib.EVP_CipherUpdate(ctx, out, ctypes.byref(n), data, len(data)) != 1):
                raise ReolinkError('OpenSSL AES-CFB failed')
            return out.raw[:n.value]
        finally:
            lib.EVP_CIPHER_CTX_free(ctx)
    return aes


def _md5_31(text):
    return hashlib.md5(text.encode()).hexdigest().upper()[:31]


# --- Native protocol client ---------------------------------------------------
class BaichuanClient:
    """One TCP session with the NVR's native protocol (the media port, 9000 by default)."""

    def __init__(self, host, port=9000, timeout=10.0):
        self.host, self.port = host, port
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except OSError:
            pass
        self._send_lock = threading.Lock()
        self._enc = 'none'
        self._key = None
        self._num = 0
        self._preview = None

    def close(self):
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    # Framing ----------------------------------------------------------------
    def _crypt(self, data, channel, encrypt, mode=None):
        mode = self._enc if mode is None else mode
        if not data or mode == 'none':
            return bytes(data)
        if mode == 'bc':
            return _bc_xor(channel, data)
        return _aes_cfb()(self._key, bytes(data), encrypt)

    def _send(self, msg_id, body, *, num, channel=0, resp=0, cls=0x6414, mode=None):
        if isinstance(body, str):
            body = body.encode()
        body = self._crypt(body, channel, True, mode)
        head = _HEADER.pack(MAGIC, msg_id, len(body), channel, 0, num, resp, cls)
        if cls in _CLASSES_WITH_PAYLOAD_OFFSET:
            head += struct.pack('<I', 0)
        with self._send_lock:
            self._sock.sendall(head + body)

    def _next_num(self):
        self._num = (self._num + 1) & 0xFFFF
        return self._num

    def _recv_exact(self, n):
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self._sock.recv_into(view[got:], n - got)
            if not k:
                raise ConnectionError('the NVR closed the connection')
            got += k
        return buf

    def read_message(self):
        magic, msg_id, length, channel, _stype, num, resp, cls = _HEADER.unpack(self._recv_exact(_HEADER.size))
        if magic not in (MAGIC, MAGIC_REV):
            raise ReolinkError(f'unexpected data from the NVR (magic {magic:#x})')
        offset = struct.unpack('<I', self._recv_exact(4))[0] if cls in _CLASSES_WITH_PAYLOAD_OFFSET else 0
        if length > _MAX_BODY:
            raise ReolinkError(f'message too large ({length} bytes)')
        body = self._recv_exact(length) if length else bytearray()
        return BcMessage(msg_id, channel, num, resp, cls, offset, body)

    def _read_reply(self, msg_id, num, limit=50):
        for _ in range(limit):
            msg = self.read_message()
            if msg.msg_id == msg_id and msg.num == num:
                return msg
        raise ReolinkError(f'no reply to message {msg_id}')

    # Commands ---------------------------------------------------------------
    def login(self, username, password):
        """Legacy login (for the nonce and encryption choice), then the modern login."""
        num = self._next_num()
        legacy = ((_md5_31(username) + '\0').encode().ljust(32, b'\0')
                  + (_md5_31(password) + '\0').encode().ljust(32, b'\0')).ljust(1836, b'\0')
        self._send(MSG_LOGIN, legacy, num=num, resp=0xDC03, cls=0x6514, mode='none')
        reply = self._read_reply(MSG_LOGIN, num)
        enc = _ENCRYPTION.get(reply.resp)
        if enc is None:
            raise LoginFailed(f'unexpected login reply from the NVR ({reply.resp:#x})')
        payload = reply.body[reply.payload_offset:]
        nonce = re.search(rb'<nonce>([^<]+)</nonce>', _bc_xor(reply.channel, payload))
        if not nonce:
            raise LoginFailed('the NVR sent no login nonce')
        nonce = nonce.group(1).decode()
        if enc in ('aes', 'fullaes'):
            if _aes_cfb() is None:
                raise ReolinkError('AES is needed but neither `cryptography` nor OpenSSL libcrypto was found')
            self._key = hashlib.md5(f'{nonce}-{password}'.encode()).hexdigest().upper()[:16].encode()
        # The modern login itself is XOR-encoded; the negotiated encryption applies afterwards.
        self._send(MSG_LOGIN, _LOGIN_XML.format(user=_md5_31(username + nonce), password=_md5_31(password + nonce)),
                   num=num, mode='none' if enc == 'none' else 'bc')
        reply = self._read_reply(MSG_LOGIN, num)
        if reply.resp != 200:
            raise LoginFailed('wrong NVR username or password' if reply.resp == 401
                              else f'the NVR refused the login (code {reply.resp})')
        self._enc = enc

    def start_preview(self, channel, stream='mainStream'):
        xml = _PREVIEW_XML.format(channel=channel, handle=STREAM_HANDLES[stream], stream=stream)
        self._preview = (channel, xml)
        self._send(MSG_VIDEO, xml, num=self._next_num(), channel=channel)

    def stop_preview(self):
        if self._preview:
            channel, xml = self._preview
            self._preview = None
            self._send(MSG_VIDEO_STOP, xml, num=self._next_num(), channel=channel)

    def iter_media(self):
        """Yield the raw media bytes (BcMedia) of the running preview."""
        binary = set()
        while True:
            msg = self.read_message()
            if msg.msg_id != MSG_VIDEO:
                continue
            if msg.resp >= 300:
                raise StreamRefused(msg.resp)
            body = memoryview(msg.body)
            payload = body[msg.payload_offset:]
            if msg.payload_offset:
                ext = self._crypt(body[:msg.payload_offset], msg.channel, False)
                if b'binaryData>1' in ext:
                    binary.add(msg.num)
                enc_len = re.search(rb'<encryptLen>(\d+)</encryptLen>', ext)
                if enc_len and self._enc == 'fullaes':
                    n = int(enc_len.group(1))
                    payload = memoryview(self._crypt(payload[:n], msg.channel, False) + bytes(payload[n:]))
            if msg.num in binary and len(payload):
                yield payload


class BcMediaParser:
    """Splits the NVR's media byte stream ("BcMedia") into info and video packets."""

    _RESYNC = re.compile(rb'[0-9][01]dcH26[45]|100[12]|0[15]wb')

    def __init__(self):
        self._buf = bytearray()
        self._pos = 0
        self.resyncs = 0

    def feed(self, data):
        """Add bytes; return [('info', w, h, fps) | ('video', codec, keyframe, payload, micros)]."""
        buf = self._buf
        buf += data
        pos = self._pos
        events = []
        while len(buf) - pos >= 8:
            magic = bytes(buf[pos:pos + 4])
            if magic in (b'1001', b'1002'):  # stream info
                size = struct.unpack_from('<I', buf, pos + 4)[0]
                if not 16 <= size <= 1024:
                    pos = self._resync(pos)
                    continue
                if len(buf) - pos < size:
                    break
                width, height = struct.unpack_from('<II', buf, pos + 8)
                events.append(('info', width, height, buf[pos + 17]))
                pos += size
            elif magic[2:] == b'dc' and magic[1:2] in (b'0', b'1') and 48 <= magic[0] <= 57:  # I / P frame
                if len(buf) - pos < 24:
                    break
                codec = bytes(buf[pos + 4:pos + 8])
                size, extra, micros = struct.unpack_from('<III', buf, pos + 8)
                if codec not in (b'H264', b'H265') or size > 32_000_000 or extra > 1024:
                    pos = self._resync(pos)
                    continue
                total = 24 + extra + size + (-size % 8)
                if len(buf) - pos < total:
                    break
                start = pos + 24 + extra
                events.append(('video', 'hevc' if codec == b'H265' else 'h264', magic[1:2] == b'0',
                               bytes(buf[start:start + size]), micros))
                pos += total
            elif magic in (b'05wb', b'01wb'):  # AAC / ADPCM audio: skipped
                size = struct.unpack_from('<H', buf, pos + 4)[0]
                total = 8 + size + (-size % 8)
                if len(buf) - pos < total:
                    break
                pos += total
            else:
                pos = self._resync(pos)
        if pos > (1 << 20):
            del buf[:pos]
            pos = 0
        self._pos = pos
        return events

    def _resync(self, pos):
        self.resyncs += 1
        match = self._RESYNC.search(self._buf, pos + 1)
        return match.start() if match else max(pos + 1, len(self._buf) - 7)


# --- Frame conversion ---------------------------------------------------------
@functools.lru_cache(maxsize=8)
def _ycc_to_bgr_matrix(colorspace, full_range):
    """3x4 matrix taking [Y, Cb, Cr, 1] to [B, G, R] for cv2.transform."""
    kr, kb = (0.2126, 0.0722) if colorspace == 1 else (0.299, 0.114)  # BT.709, else BT.601
    kg = 1.0 - kr - kb
    ys, y0, cs = (1.0, 0.0, 1.0) if full_range else (255 / 219, 16.0, 255 / 224)
    b_cb, r_cr = 2 * (1 - kb) * cs, 2 * (1 - kr) * cs
    g_cb, g_cr = -2 * (1 - kb) * kb / kg * cs, -2 * (1 - kr) * kr / kg * cs
    return np.array([[ys, b_cb, 0.0, -ys * y0 - b_cb * 128],
                     [ys, g_cb, g_cr, -ys * y0 - (g_cb + g_cr) * 128],
                     [ys, 0.0, r_cr, -ys * y0 - r_cr * 128]], np.float32)


def _plane(plane, rows, cols, channels=1):
    arr = np.frombuffer(plane, np.uint8, count=plane.line_size * plane.height).reshape(plane.height, plane.line_size)
    arr = arr[:rows, :cols * channels]
    return arr.reshape(rows, cols, channels) if channels > 1 else arr


def frame_to_bgr(frame, rect=None, size=None):
    """BGR image of (part of) a decoded frame.

    Crops in YUV before scaling and colour conversion, so a zoomed-in view of a
    4K frame costs a fraction of converting the whole frame. rect = (x, y, w, h)
    in frame pixels, size = (width, height) of the output.
    """
    W, H = frame.width, frame.height
    if rect is None:
        x, y, w, h = 0, 0, W, H
    else:
        x, y, w, h = (int(v) for v in rect)
        x = min(max(0, x), W - 2) & ~1
        y = min(max(0, y), H - 2) & ~1
        w = max(2, min(w, W - x)) & ~1
        h = max(2, min(h, H - y)) & ~1
    ow, oh = (w, h) if size is None else (max(1, int(size[0])), max(1, int(size[1])))
    fmt = frame.format.name
    if fmt not in ('nv12', 'yuv420p', 'yuvj420p'):  # e.g. 10-bit video: let FFmpeg convert
        img = frame.to_ndarray(format='bgr24')[y:y + h, x:x + w]
        if (ow, oh) == (w, h):
            return np.ascontiguousarray(img)
        return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA if ow < w else cv2.INTER_LINEAR)

    luma = _plane(frame.planes[0], H, W)[y:y + h, x:x + w]
    cy, cx, ch, cw = y // 2, x // 2, h // 2, w // 2
    luma = luma if (ow, oh) == (w, h) else cv2.resize(
        luma, (ow, oh), interpolation=cv2.INTER_AREA if ow < w else cv2.INTER_LINEAR)
    chroma_interp = cv2.INTER_AREA if ow < cw else cv2.INTER_LINEAR
    if fmt == 'nv12':
        uv = cv2.resize(_plane(frame.planes[1], H // 2, W // 2, 2)[cy:cy + ch, cx:cx + cw], (ow, oh),
                        interpolation=chroma_interp)
        cb, cr = uv[..., 0], uv[..., 1]
    else:
        cb = cv2.resize(_plane(frame.planes[1], H // 2, W // 2)[cy:cy + ch, cx:cx + cw], (ow, oh),
                        interpolation=chroma_interp)
        cr = cv2.resize(_plane(frame.planes[2], H // 2, W // 2)[cy:cy + ch, cx:cx + cw], (ow, oh),
                        interpolation=chroma_interp)
    ycc = cv2.merge([np.ascontiguousarray(luma), np.ascontiguousarray(cb), np.ascontiguousarray(cr)])
    full_range = frame.color_range == 2 or fmt == 'yuvj420p'
    return cv2.transform(ycc, _ycc_to_bgr_matrix(int(frame.colorspace or 0), full_range))


# --- Decoding -----------------------------------------------------------------
def _default_hw_devices():
    try:
        from av.codec.hwaccel import hwdevices_available
        available = set(hwdevices_available())
    except Exception:
        return []
    preferred = {'win32': ['d3d11va', 'dxva2'], 'darwin': ['videotoolbox']}.get(sys.platform, ['cuda', 'vaapi'])
    return [d for d in preferred if d in available]


def open_decoder(codec, decoder='auto'):
    """(CodecContext, requested device) for codec 'hevc'/'h264'.

    decoder: 'auto' (GPU if available, else software), 'software', or an FFmpeg
    hardware device type such as 'd3d11va', 'cuda', 'vaapi', 'qsv'. A GPU decoder
    falls back to software if the device can't be opened.
    """
    devices = [] if decoder == 'software' else (_default_hw_devices() if decoder == 'auto' else [decoder])
    for device in devices:
        try:
            from av.codec.hwaccel import HWAccel
            return av.CodecContext.create(codec, 'r', hwaccel=HWAccel(device_type=device,
                                                                      allow_software_fallback=True)), device
        except Exception as e:
            print(f'[4K] Could not open {device} decoder ({e}); trying the next option')
    return av.CodecContext.create(codec, 'r'), 'software'


class NativeStream:
    """Keeps one channel's stream running and decoded in a background thread.

    latest() returns the newest decoded av.VideoFrame; convert it with
    frame_to_bgr(), cropping first. Reconnects on its own until stop().
    """

    def __init__(self, host, username, password, channel, *, port=9000, stream='mainStream', decoder='auto'):
        self.host, self.port = host, port
        self._username, self._password = username, password
        self.channel, self.stream, self.decoder_pref = channel, stream, decoder
        self.status = 'idle'
        self.error = None
        self.codec = self.decoder = None
        self.width = self.height = self.camera_fps = None
        self.reconnects = 0
        self.decode_errors = 0
        self._client = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._frame, self._frame_id, self._frame_time = None, 0, 0.0
        self._frame_times = deque(maxlen=120)
        self._bytes = deque(maxlen=400)
        self._attempts = 0
        self._refusals = 0
        self._thread = threading.Thread(target=self._run, name=f'reolink-4k-ch{channel}', daemon=True)

    # Public API -------------------------------------------------------------
    def start(self):
        self._thread.start()
        return self

    def stop(self, timeout=3.0):
        self._stop.set()
        client = self._client
        if client:
            try:
                client.stop_preview()
            except OSError:
                pass
            client.close()
        if timeout and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    def wait_ready(self, timeout):
        """True once the first frame is decoded; False if the stream fails first or times out."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._ready.wait(0.05):
                return True
            if self.status == 'login_failed' or (self.status == 'refused' and self._refusals >= 3):
                return False
            if self.status == 'error' and self._attempts >= 2:
                return False
        return self._ready.is_set()

    def latest(self):
        """(frame id, av.VideoFrame or None, wall-clock time it was decoded)."""
        with self._lock:
            return self._frame_id, self._frame, self._frame_time

    def info(self):
        now = time.monotonic()
        times = [t for t in list(self._frame_times) if now - t <= 3.0]  # list(): the decoder thread appends
        fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 2 and times[-1] > times[0] else 0.0
        recent = [(t, n) for t, n in list(self._bytes) if now - t <= 3.0]
        span = (now - recent[0][0]) if recent else 0
        mbps = sum(n for _, n in recent) * 8 / span / 1e6 if span > 0.5 else 0.0
        _, _, frame_time = self.latest()
        return {
            'channel': self.channel, 'status': self.status, 'error': self.error,
            'width': self.width, 'height': self.height, 'codec': self.codec, 'decoder': self.decoder,
            'fps': round(fps, 1), 'bitrate_mbps': round(mbps, 2),
            'frame_age_ms': int((time.time() - frame_time) * 1000) if frame_time else None,
            'reconnects': self.reconnects, 'decode_errors': self.decode_errors,
        }

    # Worker -----------------------------------------------------------------
    def _set_status(self, status, error=None):
        if error:
            error = error.replace(self._password, '***') if self._password else error
        if (status, error) != (self.status, self.error):
            print(f'[4K] Camera {self.channel + 1}: {status}{f" - {error}" if error else ""}')
        self.status, self.error = status, error

    def _run(self):
        while not self._stop.is_set():
            self._attempts += 1
            delay = 2.0
            try:
                self._stream_once()
            except StreamRefused as e:
                self._refusals += 1
                self._set_status('refused', f'The NVR refused the 4K stream (code {e.code}). It serves at most two '
                                            '4K streams at once; another app may be using one.')
                delay = 1.0 if self._refusals < 3 else 10.0  # the NVR frees a slot a moment after a stop
            except LoginFailed as e:
                self._set_status('login_failed', str(e))
                delay = 30.0
            except (OSError, ReolinkError, av.FFmpegError, ValueError) as e:
                if not self._stop.is_set():
                    self._set_status('error', f'{type(e).__name__}: {e}')
                delay = min(2.0 * self._attempts, 15.0)
            finally:
                client, self._client = self._client, None
                if client:
                    client.close()
            if self._stop.wait(delay):
                break
            self.reconnects += 1
        self._set_status('stopped')

    def _stream_once(self):
        self._set_status('connecting')
        client = self._client = BaichuanClient(self.host, self.port)
        if self._stop.is_set():
            return
        client.login(self._username, self._password)
        client.start_preview(self.channel, self.stream)
        self._set_status('starting')
        parser = BcMediaParser()
        decoder = codec = requested = None
        for chunk in client.iter_media():
            if self._stop.is_set():
                return
            for event in parser.feed(chunk):
                if event[0] == 'info':
                    self.width, self.height, self.camera_fps = event[1:]
                    continue
                _, name, keyframe, data, _micros = event
                self._bytes.append((time.monotonic(), len(data)))
                if decoder is None or name != codec:
                    if not keyframe:
                        continue  # start decoding at a keyframe
                    decoder, requested = open_decoder(name, self.decoder_pref)
                    codec = name
                    self.codec = 'H.265' if name == 'hevc' else 'H.264'
                    self.decoder = None
                try:
                    frames = decoder.decode(av.Packet(data))
                except av.FFmpegError:
                    self.decode_errors += 1
                    continue
                for frame in frames:
                    if self.decoder is None:
                        self.decoder = requested if getattr(decoder, 'is_hwaccel', False) else 'software'
                    with self._lock:
                        self._frame, self._frame_time = frame, time.time()
                        self._frame_id += 1
                    self._frame_times.append(time.monotonic())
                    self.width, self.height = frame.width, frame.height
                    if self.status != 'streaming':
                        self._refusals = 0
                        self._set_status('streaming')
                        self._ready.set()


class HDStreamManager:
    """Starts 4K streams on demand, at most `max_streams` at once, and stops idle ones."""

    def __init__(self, host, username, password, *, port=9000, max_streams=2, idle_seconds=20.0,
                 decoder='auto', stream='mainStream', start_timeout=8.0):
        self.host, self.port = host, port
        self._username, self._password = username, password
        self.max_streams, self.idle_seconds = max_streams, idle_seconds
        self.decoder, self.stream, self.start_timeout = decoder, stream, start_timeout
        self._streams = {}
        self._last_used = {}
        self._lock = threading.Lock()
        self._reaper = None
        self._closed = threading.Event()

    def acquire(self, channel):
        """Start (or keep) the 4K stream of `channel` and wait for its first frame.

        Raises HDLimitReached when the cap is reached, HDUnavailable when the
        stream doesn't start.
        """
        with self._lock:
            stream = self._streams.get(channel)
            if stream is None:
                if len(self._streams) >= self.max_streams:
                    active = sorted(self._streams)
                    names = ' and '.join(f'camera {c + 1}' for c in active)
                    raise HDLimitReached(f'At most {self.max_streams} cameras can stream 4K at once (the NVR\'s '
                                         f'limit). Turn 4K off on {names} first.', active)
                stream = NativeStream(self.host, self._username, self._password, channel, port=self.port,
                                      stream=self.stream, decoder=self.decoder).start()
                self._streams[channel] = stream
                if self._reaper is None:
                    self._reaper = threading.Thread(target=self._reap, name='reolink-4k-reaper', daemon=True)
                    self._reaper.start()
            self._last_used[channel] = time.monotonic()
        if not stream.wait_ready(self.start_timeout):
            info = stream.info()
            self.release(channel)
            raise HDUnavailable(info['error'] or f'No 4K video from camera {channel + 1} within '
                                                 f'{self.start_timeout:.0f} s.')
        return stream.info()

    def release(self, channel):
        with self._lock:
            stream = self._streams.pop(channel, None)
            self._last_used.pop(channel, None)
        if stream:
            stream.stop()
        return stream is not None

    def get(self, channel, touch=True):
        """The running stream of `channel`, or None. touch=True counts as use (keeps it alive)."""
        with self._lock:
            stream = self._streams.get(channel)
            if stream and touch:
                self._last_used[channel] = time.monotonic()
            return stream

    def status(self):
        with self._lock:
            streams = dict(self._streams)
        return {'max_streams': self.max_streams, 'idle_seconds': self.idle_seconds, 'decoder': self.decoder,
                'streams': {str(ch): s.info() for ch, s in sorted(streams.items())}}

    def shutdown(self):
        self._closed.set()
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
            self._last_used.clear()
        for stream in streams:
            stream.stop(timeout=1.0)

    def _reap(self):
        while not self._closed.wait(2.0):
            now = time.monotonic()
            with self._lock:
                idle = [ch for ch, t in self._last_used.items() if now - t > self.idle_seconds]
            for channel in idle:
                print(f'[4K] Camera {channel + 1}: no viewers for {self.idle_seconds:.0f} s, stopping its 4K stream')
                self.release(channel)


# --- Stills over HTTP -----------------------------------------------------------
class NvrSnapshotClient:
    """Full-resolution JPEG stills through the NVR's HTTP API (cmd=Snap)."""

    def __init__(self, host, username, password, timeout=15.0):
        self.host, self.timeout = host, timeout
        self._username, self._password = username, password
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._token, self._expires = None, 0.0

    def _api(self, cmd, param=None, token=None):
        url = f'http://{self.host}/api.cgi?cmd={cmd}' + (f'&token={token}' if token else '')
        r = self._session.post(url, json=[{'cmd': cmd, 'action': 0, 'param': param or {}}], timeout=self.timeout)
        r.raise_for_status()
        return r.json()[0]

    def _get_token(self):
        with self._lock:
            if not self._token or time.monotonic() > self._expires:
                reply = self._api('Login', {'User': {'Version': '0', 'userName': self._username,
                                                     'password': self._password}})
                if reply.get('code') != 0:
                    detail = reply.get('error', {}).get('detail', 'unknown error')
                    raise LoginFailed(f'NVR login failed: {detail}')
                token = reply['value']['Token']
                self._token = token['name']
                self._expires = time.monotonic() + int(token.get('leaseTime', 3600)) - 60
            return self._token

    def snapshot(self, channel):
        """JPEG bytes at the camera's full resolution."""
        detail = 'no image'
        for _ in range(2):
            token = self._get_token()
            r = self._session.get(f'http://{self.host}/api.cgi', timeout=self.timeout,
                                  params={'cmd': 'Snap', 'channel': channel, 'rs': uuid.uuid4().hex[:8],
                                          'token': token})
            if r.ok and r.headers.get('Content-Type', '').startswith('image/'):
                return r.content
            try:
                detail = r.json()[0].get('error', {}).get('detail', detail)
            except (ValueError, IndexError, KeyError, AttributeError):
                detail = f'HTTP {r.status_code}'
            with self._lock:  # token expired or invalid: log in again once
                self._token = None
        raise ReolinkError(f'the NVR returned no snapshot ({detail})')

    def close(self):
        with self._lock:
            token, self._token = self._token, None
        if token:
            try:
                self._api('Logout', token=token)
            except (requests.RequestException, ValueError):
                pass


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    ch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    s = NativeStream(os.getenv('REOLINK_IP'), os.getenv('REOLINK_USERNAME', 'admin'), os.getenv('REOLINK_PASSWORD'),
                     ch, port=int(os.getenv('REOLINK_MEDIA_PORT', '9000')),
                     decoder=os.getenv('REOLINK_HD_DECODER', 'auto')).start()
    try:
        if not s.wait_ready(10):
            sys.exit(f'No video: {s.info()}')
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            time.sleep(2)
            print(s.info())
        _, frame, _ = s.latest()
        out = f'camera{ch + 1}_4k.jpg'
        cv2.imwrite(out, frame_to_bgr(frame), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f'Saved {frame.width}x{frame.height} still to {out}')
    finally:
        s.stop()
