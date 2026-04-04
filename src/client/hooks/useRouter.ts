import { useCallback, useEffect, useState } from "react";

export function useRouter() {
  const [pathname, setPathname] = useState(window.location.pathname);

  useEffect(() => {
    const onPopState = () => setPathname(window.location.pathname);
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((path: string) => {
    if (path !== window.location.pathname) {
      history.pushState(null, "", path);
      window.dispatchEvent(new PopStateEvent("popstate"));
    }
  }, []);

  return { pathname, navigate };
}
