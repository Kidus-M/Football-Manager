import { Suspense } from "react";
import { CompareView } from "./CompareView";

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="p-8 text-sm text-fog">Loading players…</div>}>
      <CompareView />
    </Suspense>
  );
}
