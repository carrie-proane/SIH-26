import { describe, expect, it } from "vitest";

import { parseCsv } from "./csv";

describe("parseCsv", () => {
  it("keeps quoted camera filenames intact", () => {
    const rows = parseCsv('image_id,image_name,sfm_x\n1,"frame,0001.jpg",2.5\n');
    expect(rows).toEqual([{ image_id: "1", image_name: "frame,0001.jpg", sfm_x: "2.5" }]);
  });

  it("handles CRLF files from Windows tools", () => {
    expect(parseCsv("frame_index,timestamp_s\r\n4,1.25\r\n")).toEqual([
      { frame_index: "4", timestamp_s: "1.25" },
    ]);
  });
});
