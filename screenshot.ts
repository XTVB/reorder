import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:4928";

async function main() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  // 1. Test URL redirect from /
  await page.goto(BASE);
  await page.waitForTimeout(2000);
  const redirectedUrl = page.url();
  console.log(`/ redirected to: ${redirectedUrl}`);
  await page.screenshot({ path: "screenshots/01-reorder.png", fullPage: false });

  // 2. Navigate to cluster via URL
  await page.goto(`${BASE}/cluster`);
  await page.waitForTimeout(3000);
  await page.screenshot({ path: "screenshots/02-cluster-loaded.png", fullPage: false });

  // 3. Click toggle to go back to reorder
  const reorderBtn = await page.locator(".mode-toggle-btn").first();
  await reorderBtn.click();
  await page.waitForTimeout(1500);
  const afterToggle = page.url();
  console.log(`After toggle click: ${afterToggle}`);
  await page.screenshot({ path: "screenshots/03-toggle-back.png", fullPage: false });

  // 4. Browser back should go to cluster
  await page.goBack();
  await page.waitForTimeout(1500);
  const afterBack = page.url();
  console.log(`After browser back: ${afterBack}`);
  await page.screenshot({ path: "screenshots/04-browser-back.png", fullPage: false });

  // 5. Cluster scrolled
  await page.evaluate(() => {
    const el = document.querySelector(".cluster-list");
    if (el) el.scrollTop = 1200;
  });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: "screenshots/05-cluster-deep-scroll.png", fullPage: false });

  // 6. Reorder scrolled
  await page.goto(`${BASE}/reorder`);
  await page.waitForTimeout(2000);
  const scrollContainer = await page.$(".grid-scroll-container");
  if (scrollContainer) {
    await scrollContainer.evaluate((el: Element) => { el.scrollTop = 1600; });
    await page.waitForTimeout(1000);
  }
  await page.screenshot({ path: "screenshots/06-reorder-deep-scroll.png", fullPage: false });

  await browser.close();
  console.log("Screenshots saved to screenshots/");
}

main().catch(console.error);
