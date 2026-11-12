// A picture of what the person sees in a tab, for a model that can look at images. Only the visible part, only of
// the tab in front of its window, and only when the person turned it on — then scaled down, because a model needs
// the layout, not every retina pixel, and every pixel costs prefill time on a laptop.

import { api } from "../browser-adapters/api";

const MAX_WIDTH = 1280;

export class CaptureError extends Error {}

/** A JPEG data URL of the tab's visible area. Throws CaptureError with a reason a person can act on. */
export async function captureTab(tabId: number): Promise<string> {
  return shrink(await captureRaw(tabId));
}

/** The screenshot, scaled down, and the page's pictures cut from the full-resolution capture before scaling —
 *  so a small picture in a wide window keeps its detail. `rects` are CSS pixels; `dpr` the page's pixel ratio. */
export async function captureWithPictures(tabId: number, rects: { x: number; y: number; w: number; h: number }[],
                                          dpr: number): Promise<{ screenshot: string; pictures: string[] }> {
  const raw = await captureRaw(tabId);
  const pictures: string[] = [];
  const g = globalThis as unknown as { OffscreenCanvas?: typeof OffscreenCanvas; createImageBitmap?: typeof createImageBitmap };
  if (rects.length && g.OffscreenCanvas && g.createImageBitmap) {
    const bmp = await g.createImageBitmap(await (await fetch(raw)).blob());
    // the capture's own scale: normally the page's pixel ratio, measured from the image in case zoom differs
    const scale = dpr > 0 ? dpr : 1;
    for (const r of rects) {
      const sx = Math.round(r.x * scale), sy = Math.round(r.y * scale);
      const sw = Math.min(bmp.width - sx, Math.round(r.w * scale)), sh = Math.min(bmp.height - sy, Math.round(r.h * scale));
      if (sw < 32 || sh < 32) continue;
      const k = Math.min(1, PICTURE_MAX / Math.max(sw, sh));
      const canvas = new g.OffscreenCanvas(Math.round(sw * k), Math.round(sh * k));
      canvas.getContext("2d")!.drawImage(bmp, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
      pictures.push(await toDataUrl(await canvas.convertToBlob({ type: "image/jpeg", quality: 0.8 })));
    }
    bmp.close();
  }
  return { screenshot: await shrink(raw), pictures };
}

const PICTURE_MAX = 1024;         // the longest side of a picture cut from the page

async function toDataUrl(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return `data:${blob.type || "image/jpeg"};base64,` + btoa(bin);
}

async function captureRaw(tabId: number): Promise<string> {
  const tab = await api.tabs.get(tabId) as { windowId?: number; active?: boolean };
  // the browser can only photograph the tab in front: a background tab would give a picture of another page
  if (tab.active === false) throw new CaptureError("the tab is not in front of its window, so it cannot be seen");
  let raw: string;
  try {
    raw = await api.tabs.captureVisibleTab(tab.windowId, { format: "jpeg", quality: 75 });
  } catch (e) {
    throw new CaptureError("the browser did not allow a screenshot of this page — allow screenshots in the panel "
      + `(${e instanceof Error ? e.message : String(e)})`);
  }
  return raw;
}

async function shrink(dataUrl: string): Promise<string> {
  const g = globalThis as unknown as { OffscreenCanvas?: typeof OffscreenCanvas; createImageBitmap?: typeof createImageBitmap };
  if (!g.OffscreenCanvas || !g.createImageBitmap) return dataUrl;           // no canvas here: send it as taken
  const bmp = await g.createImageBitmap(await (await fetch(dataUrl)).blob());
  if (bmp.width <= MAX_WIDTH) { bmp.close(); return dataUrl; }
  const w = MAX_WIDTH, h = Math.round(bmp.height * (MAX_WIDTH / bmp.width));
  const canvas = new g.OffscreenCanvas(w, h);
  canvas.getContext("2d")!.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.72 });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return "data:image/jpeg;base64," + btoa(bin);
}
