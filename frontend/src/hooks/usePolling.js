import { useEffect, useRef } from 'react';

export function usePolling(fetchFn, intervalMs = 2000, enabled = true) {
  const savedFn = useRef(fetchFn);

  useEffect(() => {
    savedFn.current = fetchFn;
  }, [fetchFn]);

  useEffect(() => {
    if (!enabled) return;

    savedFn.current();
    const id = setInterval(() => savedFn.current(), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, enabled]);
}
