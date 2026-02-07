/**
 * RRG Scanner Preset Configuration - Updated Feb 8, 2026
 * Added IJR for Small-Cap Quality vs. Quantity comparison
 */

const rrgPresets = {
  // Group 1: The "Pulse" - Global Macro & Hard Assets
  majorThemes: {
    benchmark: "SPY",
    tickers: [
      "SPY", "QQQ", "DIA", "MAGS",         // The Big Four
      "IWM", "IJR",                        // Small Caps: Broad vs. Quality
      "GLD", "SLV", "COPX", "XLE", "BTC"   // Hard Assets & Crypto
    ],
    description: "Broad market leadership vs. Commodities & Small-Cap quality"
  },

  // Group 2: The Sector Rotation - Institutional Flow
  sectorRotation: {
    benchmark: "SPY",
    tickers: ["XLK", "XLY", "XLC", "XBI", "XLF", "XLI", "XLE", "XLV", "XLP", "XLU", "XLB", "XLRE"],
    description: "Offensive vs. Defensive sector strength"
  }
};

// Function remains the same to handle the updated array
function loadScannerGroup(groupId) {
  const group = rrgPresets[groupId];
  if (!group) return console.error("Group not found");

  console.log(`Analyzing: ${group.description}`);
  updateRRGChart(group.benchmark, group.tickers);
}
