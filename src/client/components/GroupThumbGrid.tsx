import React, { useMemo } from "react";
import { imageUrl } from "../utils/helpers.ts";

export const GroupThumbGrid = React.memo(function GroupThumbGrid({ images }: { images: string[] }) {
  const thumbs = useMemo(() => {
    const n = images.length;
    if (n === 0) return [];
    if (n <= 4) return images.slice();
    return [
      images[0],
      images[Math.floor(n / 3)],
      images[Math.floor((n * 2) / 3)],
      images[n - 1],
    ];
  }, [images]);

  return (
    <div className="group-thumb-grid">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="group-thumb-slot">
          {thumbs[i] ? (
            <img
              src={imageUrl(thumbs[i])}
              alt=""
              loading="lazy"
              draggable={false}
            />
          ) : (
            <div className="group-thumb-empty" />
          )}
        </div>
      ))}
    </div>
  );
});
