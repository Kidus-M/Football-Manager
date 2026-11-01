import { Suspense } from "react";
import { SimilarView } from "./SimilarView";

export default function SimilarPage() {
  return (
    <Suspense fallback={<div className="p-8 text-sm text-fog">Loading the map…</div>}>
      <SimilarView />
    </Suspense>
  );
}
