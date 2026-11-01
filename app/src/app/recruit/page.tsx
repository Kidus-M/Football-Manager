import { Suspense } from "react";
import { RecruitView } from "./RecruitView";

export default function RecruitPage() {
  return (
    <Suspense fallback={<div className="p-8 text-sm text-fog">Loading the shortlist…</div>}>
      <RecruitView />
    </Suspense>
  );
}
