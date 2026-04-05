import React, { useMemo } from "react";
import { imageUrl, pickThumbSamples } from "../utils/helpers.ts";

export const GroupThumbGrid = React.memo(function GroupThumbGrid({ images }: { images: string[] }) {
  const thumbs = useMemo(() => pickThumbSamples(images), [images]);

  return (
    <div className="group-thumb-grid">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="group-thumb-slot">
          {thumbs[i] ? (
            <img src={imageUrl(thumbs[i])} alt="" loading="lazy" draggable={false} />
          ) : (
            <div className="group-thumb-empty" />
          )}
        </div>
      ))}
    </div>
  );
});
