const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaBolt, FaBrain, FaChartLine, FaRobot, FaBatteryFull, FaGlobe, FaTrophy, FaArrowRight, FaCheck, FaLightbulb } = require("react-icons/fa");

// I render icons to base64 PNG for embedding
function renderIconSvg(IconComponent, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}
async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

// I define the color palette — Midnight Executive with energy accent
const NAVY = "0F1B2D";
const DARK_BLUE = "162338";
const MID_BLUE = "1E3A5F";
const ACCENT_CYAN = "00BCD4";
const ACCENT_GREEN = "4CAF50";
const ACCENT_ORANGE = "FF9800";
const WHITE = "FFFFFF";
const LIGHT_GRAY = "B0BEC5";
const VERY_LIGHT = "E0E7EE";
const SUBTLE_BG = "1A2940";

// I define reusable style helpers
const makeShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.25 });

async function createPresentation() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "EntsoDRL";
  pres.title = "AI-Powered Battery Trading System";

  // I preload icons
  const iconBolt = await iconToBase64Png(FaBolt, "#00BCD4", 256);
  const iconBrain = await iconToBase64Png(FaBrain, "#00BCD4", 256);
  const iconChart = await iconToBase64Png(FaChartLine, "#4CAF50", 256);
  const iconRobot = await iconToBase64Png(FaRobot, "#FF9800", 256);
  const iconBattery = await iconToBase64Png(FaBatteryFull, "#00BCD4", 256);
  const iconGlobe = await iconToBase64Png(FaGlobe, "#00BCD4", 256);
  const iconTrophy = await iconToBase64Png(FaTrophy, "#FFD700", 256);
  const iconArrow = await iconToBase64Png(FaArrowRight, "#00BCD4", 256);
  const iconCheck = await iconToBase64Png(FaCheck, "#4CAF50", 256);
  const iconLight = await iconToBase64Png(FaLightbulb, "#FF9800", 256);

  // ═══════════════════════════════════════════════════════
  // SLIDE 1 — Title
  // ═══════════════════════════════════════════════════════
  let s1 = pres.addSlide();
  s1.background = { color: NAVY };
  // I add a subtle accent bar at the top
  s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });
  // I add bottom accent
  s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.565, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s1.addImage({ data: iconBolt, x: 4.5, y: 0.8, w: 1, h: 1 });
  s1.addText("AI-Powered Battery\nTrading System", {
    x: 0.5, y: 1.9, w: 9, h: 1.4, fontSize: 40, fontFace: "Calibri",
    color: WHITE, bold: true, align: "center", lineSpacingMultiple: 1.1
  });
  s1.addText([
    { text: "Energy Management System", options: { fontSize: 20, color: ACCENT_CYAN, bold: true } }
  ], { x: 0.5, y: 3.3, w: 9, h: 0.6, align: "center" });
  s1.addShape(pres.shapes.RECTANGLE, { x: 3.5, y: 3.95, w: 3, h: 0.03, fill: { color: ACCENT_CYAN, transparency: 50 } });
  s1.addText([
    { text: "Battery Energy Storage", options: { fontSize: 16, color: LIGHT_GRAY } }
  ], { x: 0.5, y: 4.05, w: 9, h: 0.5, align: "center" });
  s1.addText("CONFIDENTIAL", { x: 7.5, y: 5.1, w: 2, h: 0.4, fontSize: 9, color: LIGHT_GRAY, align: "right" });

  // ═══════════════════════════════════════════════════════
  // SLIDE 2 — The Problem
  // ═══════════════════════════════════════════════════════
  let s2 = pres.addSlide();
  s2.background = { color: DARK_BLUE };
  s2.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s2.addImage({ data: iconLight, x: 0.6, y: 0.3, w: 0.45, h: 0.45 });
  s2.addText("The Challenge", { x: 1.15, y: 0.3, w: 8, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true, margin: 0 });

  // I create 4 problem cards
  const problems = [
    { icon: iconBattery, title: "60-70%", desc: "BESS without smart EMS\nloses most potential value" },
    { icon: iconChart, title: "96/day", desc: "Decisions required across\n5+ markets simultaneously" },
    { icon: iconBolt, title: "5 Markets", desc: "DAM, IDA, aFRR, XBID, mFRR\neach with unique rules" },
    { icon: iconRobot, title: "Impossible", desc: "Manual trading cannot\noptimize all markets" },
  ];
  problems.forEach((p, i) => {
    const x = 0.5 + i * 2.35;
    const y = 1.3;
    s2.addShape(pres.shapes.RECTANGLE, { x, y, w: 2.15, h: 3.5, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
    s2.addImage({ data: p.icon, x: x + 0.75, y: y + 0.3, w: 0.6, h: 0.6 });
    s2.addText(p.title, { x, y: y + 1.1, w: 2.15, h: 0.55, fontSize: 28, fontFace: "Calibri", color: ACCENT_ORANGE, bold: true, align: "center" });
    s2.addText(p.desc, { x: x + 0.15, y: y + 1.75, w: 1.85, h: 1.4, fontSize: 13, fontFace: "Calibri", color: LIGHT_GRAY, align: "center", lineSpacingMultiple: 1.3 });
  });

  // ═══════════════════════════════════════════════════════
  // SLIDE 3 — Why Human Traders Can't Keep Up
  // ═══════════════════════════════════════════════════════
  let s3a = pres.addSlide();
  s3a.background = { color: DARK_BLUE };
  s3a.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_ORANGE } });

  s3a.addText("Why Human Traders Can't Keep Up", { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 26, fontFace: "Calibri", color: WHITE, bold: true });

  // Left: Human limitations
  s3a.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 4.3, h: 4.2, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
  s3a.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 4.3, h: 0.45, fill: { color: "8B0000" } });
  s3a.addText("Human Trader", { x: 0.5, y: 1.0, w: 4.3, h: 0.45, fontSize: 13, fontFace: "Calibri", color: WHITE, bold: true, align: "center" });

  const humanLimits = [
    { stat: "5-10", unit: "variables", desc: "Maximum simultaneous factors\na human can process" },
    { stat: "2-3", unit: "markets", desc: "Can monitor effectively\nat any given time" },
    { stat: "30 sec", unit: "reaction", desc: "Minimum decision time\nunder cognitive load" },
    { stat: "8h", unit: "attention", desc: "Maximum sustained focus\nbefore fatigue errors" },
  ];
  humanLimits.forEach((h, i) => {
    const y = 1.65 + i * 0.85;
    s3a.addText(h.stat, { x: 0.7, y, w: 1.2, h: 0.4, fontSize: 22, fontFace: "Calibri", color: "FF6B6B", bold: true, margin: 0 });
    s3a.addText(h.unit, { x: 0.7, y: y + 0.38, w: 1.2, h: 0.3, fontSize: 10, fontFace: "Calibri", color: LIGHT_GRAY, margin: 0 });
    s3a.addText(h.desc, { x: 2.0, y, w: 2.6, h: 0.7, fontSize: 11, fontFace: "Calibri", color: LIGHT_GRAY, valign: "middle", lineSpacingMultiple: 1.2 });
  });

  // Right: AI capabilities
  s3a.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.3, h: 4.2, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
  s3a.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.3, h: 0.45, fill: { color: "1B5E20" } });
  s3a.addText("AI Trading Agent", { x: 5.2, y: 1.0, w: 4.3, h: 0.45, fontSize: 13, fontFace: "Calibri", color: WHITE, bold: true, align: "center" });

  const aiCaps = [
    { stat: "131+", unit: "variables", desc: "Processes all market signals\nsimultaneously" },
    { stat: "5+", unit: "markets", desc: "Optimizes all markets\nin every decision cycle" },
    { stat: "< 1 ms", unit: "reaction", desc: "Instant response to\nmarket changes" },
    { stat: "24/7", unit: "operation", desc: "No fatigue, no emotions,\nno missed opportunities" },
  ];
  aiCaps.forEach((a, i) => {
    const y = 1.65 + i * 0.85;
    s3a.addText(a.stat, { x: 5.4, y, w: 1.2, h: 0.4, fontSize: 22, fontFace: "Calibri", color: ACCENT_GREEN, bold: true, margin: 0 });
    s3a.addText(a.unit, { x: 5.4, y: y + 0.38, w: 1.2, h: 0.3, fontSize: 10, fontFace: "Calibri", color: LIGHT_GRAY, margin: 0 });
    s3a.addText(a.desc, { x: 6.7, y, w: 2.6, h: 0.7, fontSize: 11, fontFace: "Calibri", color: LIGHT_GRAY, valign: "middle", lineSpacingMultiple: 1.2 });
  });

  // ═══════════════════════════════════════════════════════
  // SLIDE 4 — The Complexity Problem
  // ═══════════════════════════════════════════════════════
  let s3b = pres.addSlide();
  s3b.background = { color: DARK_BLUE };
  s3b.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_ORANGE } });

  s3b.addImage({ data: iconBrain, x: 0.6, y: 0.25, w: 0.4, h: 0.4 });
  s3b.addText("The Complexity Problem", { x: 1.1, y: 0.25, w: 8, h: 0.5, fontSize: 26, fontFace: "Calibri", color: WHITE, bold: true, margin: 0 });

  // I show the data dimensions a trader must process
  s3b.addText("Every 15 minutes, the trading decision depends on:", {
    x: 0.5, y: 0.9, w: 9, h: 0.4, fontSize: 14, fontFace: "Calibri", color: LIGHT_GRAY, italic: true
  });

  const dataGroups = [
    { title: "Price Signals", items: "DAM price, XBID bid/ask, ISP1/2/3\naFRR capacity price, mFRR price\nImbalance price, settlement price", count: "15+", color: ACCENT_CYAN },
    { title: "System State", items: "Battery SoC, available capacity\nDAM schedule status, IDA positions\nCycles today, delivery obligations", count: "10+", color: ACCENT_GREEN },
    { title: "External Data", items: "Serbia D+1 (24h hourly prices)\nNeighbor countries (BG, RO, HU, IT)\nWeather, wind, solar, gas prices", count: "50+", color: ACCENT_ORANGE },
    { title: "Temporal Context", items: "Hour of day, day of week, season\nHours to next gate closure\nPeak/off-peak indicators", count: "10+", color: "9C27B0" },
  ];
  dataGroups.forEach((g, i) => {
    const x = 0.4 + i * 2.4;
    const y = 1.5;
    s3b.addShape(pres.shapes.RECTANGLE, { x, y, w: 2.2, h: 3.0, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
    s3b.addShape(pres.shapes.RECTANGLE, { x, y, w: 2.2, h: 0.04, fill: { color: g.color } });
    s3b.addText(g.count, { x, y: y + 0.15, w: 2.2, h: 0.45, fontSize: 24, fontFace: "Calibri", color: g.color, bold: true, align: "center" });
    s3b.addText(g.title, { x, y: y + 0.6, w: 2.2, h: 0.35, fontSize: 12, fontFace: "Calibri", color: WHITE, bold: true, align: "center" });
    s3b.addText(g.items, { x: x + 0.1, y: y + 1.05, w: 2.0, h: 1.7, fontSize: 10, fontFace: "Calibri", color: LIGHT_GRAY, lineSpacingMultiple: 1.3 });
  });

  // I add the punchline
  s3b.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 4.75, w: 7, h: 0.65, fill: { color: "0D47A1" }, shadow: makeShadow() });
  s3b.addText([
    { text: "85+ variables ", options: { fontSize: 18, color: ACCENT_ORANGE, bold: true } },
    { text: "updated every 15 minutes across ", options: { fontSize: 14, color: WHITE } },
    { text: "5 markets ", options: { fontSize: 18, color: ACCENT_ORANGE, bold: true } },
    { text: "= impossible for humans", options: { fontSize: 14, color: LIGHT_GRAY } },
  ], { x: 1.6, y: 4.78, w: 6.8, h: 0.58, align: "center", valign: "middle" });

  // ═══════════════════════════════════════════════════════
  // SLIDE 5 — Solution: 2-Layer Architecture
  // ═══════════════════════════════════════════════════════
  let s3 = pres.addSlide();
  s3.background = { color: DARK_BLUE };
  s3.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s3.addImage({ data: iconBrain, x: 0.6, y: 0.3, w: 0.45, h: 0.45 });
  s3.addText("2-Layer AI Architecture", { x: 1.15, y: 0.3, w: 8, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true, margin: 0 });

  // Layer 1 card
  s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.1, w: 4.3, h: 4.0, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
  s3.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.1, w: 4.3, h: 0.5, fill: { color: MID_BLUE } });
  s3.addText("LAYER 1: AI Price Forecasting", { x: 0.5, y: 1.1, w: 4.3, h: 0.5, fontSize: 14, fontFace: "Calibri", color: ACCENT_CYAN, bold: true, align: "center" });
  s3.addText([
    { text: "LightGBM Machine Learning", options: { fontSize: 14, color: WHITE, bold: true, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "1 model per market (DAM, IDA1, IDA2)", options: { fontSize: 12, color: LIGHT_GRAY, bullet: true, breakLine: true } },
    { text: "24h price prediction with 45 features", options: { fontSize: 12, color: LIGHT_GRAY, bullet: true, breakLine: true } },
    { text: "Information cascade architecture", options: { fontSize: 12, color: LIGHT_GRAY, bullet: true, breakLine: true } },
    { text: "LP Optimizer converts to schedule", options: { fontSize: 12, color: LIGHT_GRAY, bullet: true, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "Serbia D+1 correlation: 0.90", options: { fontSize: 13, color: ACCENT_GREEN, bold: true } },
  ], { x: 0.75, y: 1.75, w: 3.8, h: 3.2, lineSpacingMultiple: 1.2 });

  // Layer 2 card
  s3.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.1, w: 4.3, h: 4.0, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
  s3.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.1, w: 4.3, h: 0.5, fill: { color: MID_BLUE } });
  s3.addText("LAYER 2: AI Trading Agent", { x: 5.2, y: 1.1, w: 4.3, h: 0.5, fontSize: 14, fontFace: "Calibri", color: ACCENT_ORANGE, bold: true, align: "center" });
  s3.addText([
    { text: "Reinforcement Learning (PPO)", options: { fontSize: 14, color: WHITE, bold: true, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "Real-time aFRR commitment decisions", options: { fontSize: 12, color: LIGHT_GRAY, bullet: true, breakLine: true } },
    { text: "7 commitment levels (0-30 MW)", options: { fontSize: 12, color: LIGHT_GRAY, bullet: true, breakLine: true } },
    { text: "Learns WHEN and HOW MUCH", options: { fontSize: 12, color: LIGHT_GRAY, bullet: true, breakLine: true } },
    { text: "Action masking prevents errors", options: { fontSize: 12, color: LIGHT_GRAY, bullet: true, breakLine: true } },
    { text: "", options: { fontSize: 8, breakLine: true } },
    { text: "SoC-aware: protects battery", options: { fontSize: 13, color: ACCENT_GREEN, bold: true } },
  ], { x: 5.45, y: 1.75, w: 3.8, h: 3.2, lineSpacingMultiple: 1.2 });

  // ═══════════════════════════════════════════════════════
  // SLIDE 4 — Information Cascade
  // ═══════════════════════════════════════════════════════
  let s4 = pres.addSlide();
  s4.background = { color: DARK_BLUE };
  s4.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s4.addText("Information Cascade", { x: 0.5, y: 0.3, w: 9, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true });

  const cascade = [
    { time: "D-1 12:00", market: "DAM", info: "Historical prices, neighbors,\nweather forecast, gas prices", color: "1565C0", w: 5.5 },
    { time: "D-1 15:00", market: "IDA1", info: "+ DAM actual results\n+ Serbia D+1 hourly (corr 0.90)", color: "1976D2", w: 6.5 },
    { time: "D-1 22:00", market: "IDA2", info: "+ ISP1 clearing prices\n+ evening load/RES actuals", color: "1E88E5", w: 7.5 },
    { time: "Real-time", market: "AI Agent", info: "+ current SoC, live prices\n+ locked schedules, imbalance", color: "2196F3", w: 8.5 },
  ];
  cascade.forEach((c, i) => {
    const y = 1.15 + i * 1.05;
    s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: c.w, h: 0.85, fill: { color: c.color, transparency: 20 } });
    s4.addText(c.time, { x: 0.65, y, w: 1.6, h: 0.85, fontSize: 12, fontFace: "Calibri", color: ACCENT_CYAN, bold: true, valign: "middle" });
    s4.addText(c.market, { x: 2.3, y, w: 1.2, h: 0.85, fontSize: 14, fontFace: "Calibri", color: WHITE, bold: true, valign: "middle" });
    s4.addText(c.info, { x: 3.6, y, w: 5, h: 0.85, fontSize: 11, fontFace: "Calibri", color: LIGHT_GRAY, valign: "middle", lineSpacingMultiple: 1.2 });
  });

  s4.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 5.0, w: 9, h: 0.04, fill: { color: ACCENT_CYAN, transparency: 50 } });
  s4.addText("Each later market improves decisions with new data", {
    x: 0.5, y: 5.1, w: 9, h: 0.4, fontSize: 14, fontFace: "Calibri", color: ACCENT_ORANGE, italic: true, align: "center"
  });

  // ═══════════════════════════════════════════════════════
  // SLIDE 5 — Results (KEY SLIDE)
  // ═══════════════════════════════════════════════════════
  let s5 = pres.addSlide();
  s5.background = { color: NAVY };
  s5.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_GREEN } });

  s5.addImage({ data: iconTrophy, x: 0.6, y: 0.3, w: 0.45, h: 0.45 });
  s5.addText("Proven Results", { x: 1.15, y: 0.3, w: 5, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true, margin: 0 });
  s5.addText("30MW / 146MWh BESS", { x: 6, y: 0.35, w: 3.5, h: 0.4, fontSize: 14, fontFace: "Calibri", color: LIGHT_GRAY, align: "right" });

  // I create the results table
  const tableRows = [
    [
      { text: "Component", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 13 } },
      { text: "EUR/day", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 13, align: "right" } },
      { text: "EUR/year", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 13, align: "right" } },
    ],
    [
      { text: "DAM Arbitrage (AI Forecasting)", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 13 } },
      { text: "6,583", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 13, align: "right" } },
      { text: "2.40M", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 13, align: "right" } },
    ],
    [
      { text: "aFRR Capacity (AI Agent)", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 13 } },
      { text: "9,500", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 13, align: "right" } },
      { text: "3.47M", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 13, align: "right" } },
    ],
    [
      { text: "TOTAL", options: { fill: { color: "0D47A1" }, color: ACCENT_GREEN, bold: true, fontSize: 15 } },
      { text: "~16,000", options: { fill: { color: "0D47A1" }, color: ACCENT_GREEN, bold: true, fontSize: 15, align: "right" } },
      { text: "~5.8M", options: { fill: { color: "0D47A1" }, color: ACCENT_GREEN, bold: true, fontSize: 15, align: "right" } },
    ],
  ];
  s5.addTable(tableRows, {
    x: 0.8, y: 1.1, w: 8.4, colW: [4.2, 2.1, 2.1],
    border: { pt: 0.5, color: MID_BLUE },
    rowH: [0.5, 0.5, 0.5, 0.55],
  });

  // I add big capture ratio
  s5.addShape(pres.shapes.RECTANGLE, { x: 2.5, y: 3.5, w: 5, h: 1.3, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
  s5.addText("94-98%", { x: 2.5, y: 3.5, w: 5, h: 0.8, fontSize: 48, fontFace: "Calibri", color: ACCENT_GREEN, bold: true, align: "center" });
  s5.addText("Capture Ratio vs Oracle (theoretical maximum)", { x: 2.5, y: 4.3, w: 5, h: 0.4, fontSize: 12, fontFace: "Calibri", color: LIGHT_GRAY, align: "center" });

  s5.addText("Backtested 2023-2026 | Greek Market (HEnEx)", {
    x: 0.5, y: 5.1, w: 9, h: 0.35, fontSize: 10, fontFace: "Calibri", color: LIGHT_GRAY, align: "center", italic: true
  });

  // ═══════════════════════════════════════════════════════
  // SLIDE 6 — Revenue per MW (THE MONEY SLIDE)
  // ═══════════════════════════════════════════════════════
  let s5b = pres.addSlide();
  s5b.background = { color: NAVY };
  s5b.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_GREEN } });

  s5b.addText("Revenue per MW", { x: 0.5, y: 0.25, w: 5, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true });
  s5b.addText("What your BESS earns with AI", { x: 5.5, y: 0.3, w: 4, h: 0.4, fontSize: 13, fontFace: "Calibri", color: LIGHT_GRAY, align: "right", italic: true });

  // I show the big number
  s5b.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 1.0, w: 7, h: 1.8, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
  s5b.addText("195,676", { x: 1.5, y: 1.05, w: 7, h: 1.0, fontSize: 56, fontFace: "Calibri", color: ACCENT_GREEN, bold: true, align: "center" });
  s5b.addText("EUR / MW / year", { x: 1.5, y: 2.0, w: 7, h: 0.4, fontSize: 18, fontFace: "Calibri", color: LIGHT_GRAY, align: "center" });
  s5b.addText("536 EUR/MW/day  |  40 EUR/kWh/year", { x: 1.5, y: 2.4, w: 7, h: 0.3, fontSize: 12, fontFace: "Calibri", color: ACCENT_CYAN, align: "center" });

  // I show scaling table
  const scaleRows = [
    [
      { text: "BESS Size", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 12 } },
      { text: "EUR/day", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 12, align: "right" } },
      { text: "EUR/year", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 12, align: "right" } },
      { text: "Payback", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 12, align: "center" } },
    ],
    [
      { text: "5 MW / 20 MWh", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 12 } },
      { text: "2,680", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 12, align: "right" } },
      { text: "978K", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 12, align: "right" } },
      { text: "1.2 yr", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, bold: true, fontSize: 12, align: "center" } },
    ],
    [
      { text: "10 MW / 40 MWh", options: { fill: { color: NAVY }, color: WHITE, fontSize: 12 } },
      { text: "5,361", options: { fill: { color: NAVY }, color: WHITE, fontSize: 12, align: "right" } },
      { text: "1.96M", options: { fill: { color: NAVY }, color: WHITE, fontSize: 12, align: "right" } },
      { text: "1.1 yr", options: { fill: { color: NAVY }, color: ACCENT_GREEN, bold: true, fontSize: 12, align: "center" } },
    ],
    [
      { text: "30 MW / 146 MWh", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 12 } },
      { text: "16,083", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, bold: true, fontSize: 13, align: "right" } },
      { text: "5.87M", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, bold: true, fontSize: 13, align: "right" } },
      { text: "0.9 yr", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, bold: true, fontSize: 13, align: "center" } },
    ],
    [
      { text: "50 MW / 200 MWh", options: { fill: { color: NAVY }, color: WHITE, fontSize: 12 } },
      { text: "26,805", options: { fill: { color: NAVY }, color: WHITE, fontSize: 12, align: "right" } },
      { text: "9.78M", options: { fill: { color: NAVY }, color: WHITE, fontSize: 12, align: "right" } },
      { text: "0.8 yr", options: { fill: { color: NAVY }, color: ACCENT_GREEN, bold: true, fontSize: 12, align: "center" } },
    ],
  ];
  s5b.addTable(scaleRows, {
    x: 1.0, y: 3.1, w: 8, colW: [2.5, 1.8, 1.8, 1.9],
    border: { pt: 0.5, color: MID_BLUE },
    rowH: [0.4, 0.4, 0.4, 0.45, 0.4],
  });

  s5b.addText("Linear scaling: double the MW = double the revenue", {
    x: 0.5, y: 5.15, w: 9, h: 0.3, fontSize: 11, fontFace: "Calibri", color: LIGHT_GRAY, align: "center", italic: true
  });

  // ═══════════════════════════════════════════════════════
  // SLIDE 7 — ROI & Use Cases
  // ═══════════════════════════════════════════════════════
  let s5c = pres.addSlide();
  s5c.background = { color: DARK_BLUE };
  s5c.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s5c.addText("Use Cases", { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true });

  // University of Cyprus card
  s5c.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 4.3, h: 4.2, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
  s5c.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.0, w: 4.3, h: 0.5, fill: { color: "1565C0" } });
  s5c.addText("University of Cyprus", { x: 0.5, y: 1.0, w: 4.3, h: 0.5, fontSize: 14, fontFace: "Calibri", color: WHITE, bold: true, align: "center" });
  s5c.addText([
    { text: "Research Platform", options: { fontSize: 13, color: ACCENT_CYAN, bold: true, breakLine: true } },
    { text: "DRL + energy markets = high-impact publications", options: { fontSize: 11, color: LIGHT_GRAY, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Student Projects", options: { fontSize: 13, color: ACCENT_CYAN, bold: true, breakLine: true } },
    { text: "BSc/MSc theses on AI trading optimization", options: { fontSize: 11, color: LIGHT_GRAY, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Green Campus", options: { fontSize: 13, color: ACCENT_CYAN, bold: true, breakLine: true } },
    { text: "PV + BESS + AI for net-zero campus goal", options: { fontSize: 11, color: LIGHT_GRAY, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "EU Funding", options: { fontSize: 13, color: ACCENT_CYAN, bold: true, breakLine: true } },
    { text: "Horizon Europe: AI + Energy = strong proposal", options: { fontSize: 11, color: LIGHT_GRAY, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Lab Infrastructure", options: { fontSize: 13, color: ACCENT_CYAN, bold: true, breakLine: true } },
    { text: "Real-world testbed for power systems research", options: { fontSize: 11, color: LIGHT_GRAY } },
  ], { x: 0.75, y: 1.65, w: 3.8, h: 3.4, lineSpacingMultiple: 1.15 });

  // CYTA card
  s5c.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.3, h: 4.2, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
  s5c.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.0, w: 4.3, h: 0.5, fill: { color: "E65100" } });
  s5c.addText("CYTA", { x: 5.2, y: 1.0, w: 4.3, h: 0.5, fontSize: 14, fontFace: "Calibri", color: WHITE, bold: true, align: "center" });
  s5c.addText([
    { text: "Data Center Power", options: { fontSize: 13, color: ACCENT_ORANGE, bold: true, breakLine: true } },
    { text: "BESS as UPS + revenue generation asset", options: { fontSize: 11, color: LIGHT_GRAY, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Peak Shaving", options: { fontSize: 13, color: ACCENT_ORANGE, bold: true, breakLine: true } },
    { text: "Reduce demand charges by 30-50%", options: { fontSize: 11, color: LIGHT_GRAY, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Energy Arbitrage", options: { fontSize: 13, color: ACCENT_ORANGE, bold: true, breakLine: true } },
    { text: "AI buys cheap (night), sells expensive (peak)", options: { fontSize: 11, color: LIGHT_GRAY, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Grid Services Revenue", options: { fontSize: 13, color: ACCENT_ORANGE, bold: true, breakLine: true } },
    { text: "aFRR capacity payments from TSO: ~317 EUR/MW/day", options: { fontSize: 11, color: LIGHT_GRAY, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Reliability", options: { fontSize: 13, color: ACCENT_ORANGE, bold: true, breakLine: true } },
    { text: "99.999% uptime with AI + rule-based fallback", options: { fontSize: 11, color: LIGHT_GRAY } },
  ], { x: 5.45, y: 1.65, w: 3.8, h: 3.4, lineSpacingMultiple: 1.15 });

  // ═══════════════════════════════════════════════════════
  // SLIDE 8 — Risk Management
  // ═══════════════════════════════════════════════════════
  let s5d = pres.addSlide();
  s5d.background = { color: DARK_BLUE };
  s5d.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s5d.addText("Risk Management & Safety", { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true });

  const risks = [
    { title: "AI Failure", solution: "Automatic fallback to rule-based EMS. Battery continues normal operation.", color: ACCENT_GREEN },
    { title: "SoC Protection", solution: "Action masking prevents physically impossible actions. SoC always 5-95%.", color: ACCENT_GREEN },
    { title: "Market Anomaly", solution: "Agent trained on 4+ years including 2022 energy crisis. Adapts to extremes.", color: ACCENT_GREEN },
    { title: "Communication Loss", solution: "Local EMS operates autonomously. AI reconnects and resumes when available.", color: ACCENT_GREEN },
    { title: "Human Override", solution: "Manual control ALWAYS available. AI is advisory, operator has final say.", color: ACCENT_GREEN },
  ];
  risks.forEach((r, i) => {
    const y = 1.0 + i * 0.88;
    s5d.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 9, h: 0.72, fill: { color: SUBTLE_BG } });
    s5d.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.06, h: 0.72, fill: { color: r.color } });
    s5d.addText(r.title, { x: 0.8, y, w: 2.2, h: 0.72, fontSize: 14, fontFace: "Calibri", color: WHITE, bold: true, valign: "middle", margin: 0 });
    s5d.addText(r.solution, { x: 3.1, y, w: 6.2, h: 0.72, fontSize: 12, fontFace: "Calibri", color: LIGHT_GRAY, valign: "middle" });
  });

  s5d.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 5.1, w: 7, h: 0.04, fill: { color: ACCENT_GREEN, transparency: 50 } });

  // ═══════════════════════════════════════════════════════
  // SLIDE 9 — Future-Proof: Dynamic Capacity Allocation
  // ═══════════════════════════════════════════════════════
  let s5e = pres.addSlide();
  s5e.background = { color: NAVY };
  s5e.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_ORANGE } });

  s5e.addText("Future-Proof: Dynamic Capacity Allocation", { x: 0.5, y: 0.25, w: 9, h: 0.5, fontSize: 24, fontFace: "Calibri", color: WHITE, bold: true });
  s5e.addText("What happens when more BESS enter the market?", { x: 0.5, y: 0.7, w: 9, h: 0.35, fontSize: 13, fontFace: "Calibri", color: LIGHT_GRAY, italic: true });

  // I show scenario table
  const scenarioRows = [
    [
      { text: "aFRR Price\nDecline", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 11 } },
      { text: "aFRR\nEUR/day", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 11, align: "right" } },
      { text: "DAM\nEUR/day", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 11, align: "right" } },
      { text: "Total\nEUR/day", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 11, align: "right" } },
      { text: "aFRR\nAllocation", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 11, align: "center" } },
      { text: "Annual", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 11, align: "right" } },
    ],
    [
      { text: "0% (today)", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, bold: true, fontSize: 11 } },
      { text: "8,588", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 11, align: "right" } },
      { text: "2,871", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 11, align: "right" } },
      { text: "11,459", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, bold: true, fontSize: 12, align: "right" } },
      { text: "58%", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 11, align: "center" } },
      { text: "4.18M", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, bold: true, fontSize: 11, align: "right" } },
    ],
    [
      { text: "30%", options: { fill: { color: NAVY }, color: WHITE, fontSize: 11 } },
      { text: "5,556", options: { fill: { color: NAVY }, color: WHITE, fontSize: 11, align: "right" } },
      { text: "3,251", options: { fill: { color: NAVY }, color: WHITE, fontSize: 11, align: "right" } },
      { text: "8,807", options: { fill: { color: NAVY }, color: WHITE, fontSize: 11, align: "right" } },
      { text: "52%", options: { fill: { color: NAVY }, color: WHITE, fontSize: 11, align: "center" } },
      { text: "3.21M", options: { fill: { color: NAVY }, color: WHITE, fontSize: 11, align: "right" } },
    ],
    [
      { text: "50% (BESS flood)", options: { fill: { color: SUBTLE_BG }, color: ACCENT_ORANGE, bold: true, fontSize: 11 } },
      { text: "3,772", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 11, align: "right" } },
      { text: "3,455", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 11, align: "right" } },
      { text: "7,227", options: { fill: { color: SUBTLE_BG }, color: ACCENT_ORANGE, bold: true, fontSize: 12, align: "right" } },
      { text: "49%", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 11, align: "center" } },
      { text: "2.64M", options: { fill: { color: SUBTLE_BG }, color: ACCENT_ORANGE, bold: true, fontSize: 11, align: "right" } },
    ],
    [
      { text: "80% (extreme)", options: { fill: { color: NAVY }, color: LIGHT_GRAY, fontSize: 11 } },
      { text: "1,311", options: { fill: { color: NAVY }, color: LIGHT_GRAY, fontSize: 11, align: "right" } },
      { text: "3,977", options: { fill: { color: NAVY }, color: LIGHT_GRAY, fontSize: 11, align: "right" } },
      { text: "5,289", options: { fill: { color: NAVY }, color: LIGHT_GRAY, fontSize: 11, align: "right" } },
      { text: "41%", options: { fill: { color: NAVY }, color: LIGHT_GRAY, fontSize: 11, align: "center" } },
      { text: "1.93M", options: { fill: { color: NAVY }, color: LIGHT_GRAY, fontSize: 11, align: "right" } },
    ],
  ];
  s5e.addTable(scenarioRows, {
    x: 0.5, y: 1.2, w: 9, colW: [1.8, 1.3, 1.3, 1.3, 1.1, 1.2],
    border: { pt: 0.5, color: MID_BLUE },
    rowH: [0.55, 0.45, 0.45, 0.45, 0.45],
  });

  // I add the key insight box
  s5e.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.85, w: 9, h: 1.5, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
  s5e.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.85, w: 0.06, h: 1.5, fill: { color: ACCENT_GREEN } });
  s5e.addText([
    { text: "AI Auto-Adaptation", options: { fontSize: 16, color: ACCENT_GREEN, bold: true, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "As aFRR prices decline, the AI automatically shifts capacity to DAM arbitrage.", options: { fontSize: 12, color: WHITE, breakLine: true } },
    { text: "Even with 80% aFRR price collapse, the system still earns 1.93M EUR/year.", options: { fontSize: 12, color: WHITE, breakLine: true } },
    { text: "", options: { fontSize: 6, breakLine: true } },
    { text: "Greece plans 4.7 GW BESS (vs 600 MW aFRR demand). Rule-based systems will fail.", options: { fontSize: 12, color: ACCENT_ORANGE, breakLine: true } },
    { text: "AI adapts. Rules don't.", options: { fontSize: 14, color: ACCENT_GREEN, bold: true } },
  ], { x: 0.8, y: 3.9, w: 8.5, h: 1.4, lineSpacingMultiple: 1.15 });

  // ═══════════════════════════════════════════════════════
  // SLIDE 10 — AI vs Rule-Based
  // ═══════════════════════════════════════════════════════
  let s6 = pres.addSlide();
  s6.background = { color: DARK_BLUE };
  s6.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s6.addText("AI Agent vs Rule-Based EMS", { x: 0.5, y: 0.3, w: 9, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true });

  const compRows = [
    [
      { text: "Feature", options: { fill: { color: MID_BLUE }, color: ACCENT_CYAN, bold: true, fontSize: 12 } },
      { text: "Rule-Based", options: { fill: { color: MID_BLUE }, color: "FF6B6B", bold: true, fontSize: 12, align: "center" } },
      { text: "AI-Powered", options: { fill: { color: MID_BLUE }, color: ACCENT_GREEN, bold: true, fontSize: 12, align: "center" } },
    ],
    [
      { text: "DAM Trading", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 12 } },
      { text: "Fixed schedule", options: { fill: { color: SUBTLE_BG }, color: LIGHT_GRAY, fontSize: 12, align: "center" } },
      { text: "Adapts to forecast", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, fontSize: 12, align: "center" } },
    ],
    [
      { text: "aFRR Commitment", options: { fill: { color: NAVY }, color: WHITE, fontSize: 12 } },
      { text: "Always 15MW", options: { fill: { color: NAVY }, color: LIGHT_GRAY, fontSize: 12, align: "center" } },
      { text: "Learns WHEN & HOW MUCH", options: { fill: { color: NAVY }, color: ACCENT_GREEN, fontSize: 12, align: "center" } },
    ],
    [
      { text: "New Market", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 12 } },
      { text: "Rewrite rules", options: { fill: { color: SUBTLE_BG }, color: LIGHT_GRAY, fontSize: 12, align: "center" } },
      { text: "Learns automatically", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, fontSize: 12, align: "center" } },
    ],
    [
      { text: "Crisis Response", options: { fill: { color: NAVY }, color: WHITE, fontSize: 12 } },
      { text: "Fails", options: { fill: { color: NAVY }, color: "FF6B6B", fontSize: 12, align: "center" } },
      { text: "Adapts", options: { fill: { color: NAVY }, color: ACCENT_GREEN, fontSize: 12, align: "center" } },
    ],
    [
      { text: "SoC Management", options: { fill: { color: SUBTLE_BG }, color: WHITE, fontSize: 12 } },
      { text: "Hard thresholds", options: { fill: { color: SUBTLE_BG }, color: LIGHT_GRAY, fontSize: 12, align: "center" } },
      { text: "Learned optimal behavior", options: { fill: { color: SUBTLE_BG }, color: ACCENT_GREEN, fontSize: 12, align: "center" } },
    ],
  ];
  s6.addTable(compRows, {
    x: 0.8, y: 1.1, w: 8.4, colW: [2.8, 2.8, 2.8],
    border: { pt: 0.5, color: MID_BLUE },
    rowH: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
  });

  // ═══════════════════════════════════════════════════════
  // SLIDE 7 — Real-Time Visualization
  // ═══════════════════════════════════════════════════════
  let s7 = pres.addSlide();
  s7.background = { color: DARK_BLUE };
  s7.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s7.addText("Real-Time Monitoring Dashboard", { x: 0.5, y: 0.3, w: 9, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true });

  // I simulate a dashboard layout with cards
  const dashItems = [
    { x: 0.5, y: 1.1, w: 2.8, h: 1.8, title: "Battery SoC", desc: "Real-time state of charge\nwith color-coded gauge\n(green/yellow/red)", icon: iconBattery },
    { x: 3.6, y: 1.1, w: 5.9, h: 1.8, title: "DAM Price Chart", desc: "Live scrolling price chart with historical context,\ncurrent position cursor, and daily mean reference line", icon: iconChart },
    { x: 0.5, y: 3.15, w: 4.4, h: 2.1, title: "P&L Tracker", desc: "Per-market revenue breakdown:\nDAM, aFRR capacity, aFRR energy,\nXBID, mFRR — with running totals\nand EUR/day + EUR/year projection", icon: iconTrophy },
    { x: 5.2, y: 3.15, w: 4.3, h: 2.1, title: "Agent Decisions", desc: "Real-time action display:\naFRR commitment level (0-30 MW)\nSoC trajectory projection\nDelivery ratio + daily cycles", icon: iconRobot },
  ];
  dashItems.forEach(d => {
    s7.addShape(pres.shapes.RECTANGLE, { x: d.x, y: d.y, w: d.w, h: d.h, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
    s7.addImage({ data: d.icon, x: d.x + 0.2, y: d.y + 0.15, w: 0.35, h: 0.35 });
    s7.addText(d.title, { x: d.x + 0.65, y: d.y + 0.12, w: d.w - 0.85, h: 0.4, fontSize: 14, fontFace: "Calibri", color: ACCENT_CYAN, bold: true, margin: 0 });
    s7.addText(d.desc, { x: d.x + 0.2, y: d.y + 0.6, w: d.w - 0.4, h: d.h - 0.75, fontSize: 11, fontFace: "Calibri", color: LIGHT_GRAY, lineSpacingMultiple: 1.3 });
  });

  // ═══════════════════════════════════════════════════════
  // SLIDE 8 — Cyprus Application
  // ═══════════════════════════════════════════════════════
  let s8 = pres.addSlide();
  s8.background = { color: DARK_BLUE };
  s8.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s8.addImage({ data: iconGlobe, x: 0.6, y: 0.3, w: 0.45, h: 0.45 });
  s8.addText("Cyprus Market Application", { x: 1.15, y: 0.3, w: 8, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true, margin: 0 });

  const cyprusItems = [
    { title: "Market-Agnostic", desc: "Same architecture, different data. Change the market feed, not the code." },
    { title: "Cyprus Markets", desc: "DAM, Balancing Market, Ancillary Services — all supported." },
    { title: "Local Data Integration", desc: "Forecasters retrained with Cyprus-specific weather, load, and price data." },
    { title: "EuroAsia Interconnector", desc: "Israel-Cyprus-Greece cable: new cross-border arbitrage opportunities." },
    { title: "Revenue Potential", desc: "Estimated 20-40 EUR/kWh/year depending on BESS size and local spreads." },
  ];
  cyprusItems.forEach((item, i) => {
    const y = 1.1 + i * 0.85;
    s8.addImage({ data: iconCheck, x: 0.7, y: y + 0.1, w: 0.3, h: 0.3 });
    s8.addText(item.title, { x: 1.2, y, w: 2.8, h: 0.65, fontSize: 14, fontFace: "Calibri", color: ACCENT_CYAN, bold: true, valign: "middle", margin: 0 });
    s8.addText(item.desc, { x: 4.0, y, w: 5.5, h: 0.65, fontSize: 12, fontFace: "Calibri", color: LIGHT_GRAY, valign: "middle" });
    if (i < cyprusItems.length - 1) {
      s8.addShape(pres.shapes.LINE, { x: 0.7, y: y + 0.75, w: 8.6, h: 0, line: { color: MID_BLUE, width: 0.5 } });
    }
  });

  // ═══════════════════════════════════════════════════════
  // SLIDE 9 — Competitive Advantages
  // ═══════════════════════════════════════════════════════
  let s9 = pres.addSlide();
  s9.background = { color: DARK_BLUE };
  s9.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s9.addText("Competitive Advantages", { x: 0.5, y: 0.3, w: 9, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true });

  const advantages = [
    { num: "01", title: "Open Architecture", desc: "No vendor lock-in. Full transparency and control." },
    { num: "02", title: "Proven Results", desc: "Backtested 2023-2026 on Greek market (HEnEx). 94% capture ratio." },
    { num: "03", title: "Multi-Market", desc: "DAM + aFRR + IDA + XBID optimized simultaneously." },
    { num: "04", title: "Continuous Learning", desc: "AI improves with every new data point. No manual rule updates." },
    { num: "05", title: "Scalable", desc: "Linear scaling with BESS size. 60MW = 2x revenue." },
  ];
  advantages.forEach((a, i) => {
    const y = 1.05 + i * 0.88;
    s9.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 0.65, h: 0.65, fill: { color: ACCENT_CYAN } });
    s9.addText(a.num, { x: 0.5, y, w: 0.65, h: 0.65, fontSize: 16, fontFace: "Calibri", color: NAVY, bold: true, align: "center", valign: "middle" });
    s9.addText(a.title, { x: 1.35, y, w: 3, h: 0.65, fontSize: 15, fontFace: "Calibri", color: WHITE, bold: true, valign: "middle", margin: 0 });
    s9.addText(a.desc, { x: 4.3, y, w: 5.2, h: 0.65, fontSize: 12, fontFace: "Calibri", color: LIGHT_GRAY, valign: "middle" });
  });

  // ═══════════════════════════════════════════════════════
  // SLIDE 10 — Next Steps
  // ═══════════════════════════════════════════════════════
  let s10 = pres.addSlide();
  s10.background = { color: NAVY };
  s10.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });
  s10.addShape(pres.shapes.RECTANGLE, { x: 0, y: 5.565, w: 10, h: 0.06, fill: { color: ACCENT_CYAN } });

  s10.addImage({ data: iconArrow, x: 0.6, y: 0.3, w: 0.45, h: 0.45 });
  s10.addText("Next Steps", { x: 1.15, y: 0.3, w: 8, h: 0.5, fontSize: 28, fontFace: "Calibri", color: WHITE, bold: true, margin: 0 });

  // I create 3 step cards
  const steps = [
    { num: "1", title: "Pilot Project", desc: "3-month proof of concept\nwith real market data\nand simulated trading", color: ACCENT_CYAN },
    { num: "2", title: "Market Adaptation", desc: "4-6 weeks to retrain\nforecasters with\nCyprus market data", color: ACCENT_GREEN },
    { num: "3", title: "Live Deployment", desc: "Full integration with\nEMS and real-time\nmarket connection", color: ACCENT_ORANGE },
  ];
  steps.forEach((st, i) => {
    const x = 0.7 + i * 3.1;
    s10.addShape(pres.shapes.RECTANGLE, { x, y: 1.2, w: 2.8, h: 2.8, fill: { color: SUBTLE_BG }, shadow: makeShadow() });
    s10.addShape(pres.shapes.RECTANGLE, { x, y: 1.2, w: 2.8, h: 0.06, fill: { color: st.color } });
    s10.addText(st.num, { x, y: 1.4, w: 2.8, h: 0.6, fontSize: 32, fontFace: "Calibri", color: st.color, bold: true, align: "center" });
    s10.addText(st.title, { x, y: 2.0, w: 2.8, h: 0.45, fontSize: 16, fontFace: "Calibri", color: WHITE, bold: true, align: "center" });
    s10.addText(st.desc, { x: x + 0.2, y: 2.55, w: 2.4, h: 1.2, fontSize: 12, fontFace: "Calibri", color: LIGHT_GRAY, align: "center", lineSpacingMultiple: 1.3 });
  });

  // I add requirements section
  s10.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 4.3, w: 8.6, h: 0.9, fill: { color: SUBTLE_BG } });
  s10.addText([
    { text: "Requirements:  ", options: { fontSize: 12, color: ACCENT_CYAN, bold: true } },
    { text: "BESS specifications (MW/MWh)  |  Local market data (DAM prices, ancillary)  |  Regulatory framework", options: { fontSize: 12, color: LIGHT_GRAY } },
  ], { x: 0.9, y: 4.35, w: 8.2, h: 0.4 });
  s10.addText([
    { text: "Contact:  ", options: { fontSize: 12, color: ACCENT_CYAN, bold: true } },
    { text: "[Your Name]  |  [email]  |  [phone]", options: { fontSize: 12, color: LIGHT_GRAY } },
  ], { x: 0.9, y: 4.75, w: 8.2, h: 0.4 });

  // I save the presentation
  await pres.writeFile({ fileName: "D:/WSLUbuntu/EntsoDRL/docs/AI_BESS_Trading_Presentation_Cyprus.pptx" });
  console.log("Presentation saved to docs/AI_BESS_Trading_Presentation_Cyprus.pptx");
}

createPresentation().catch(err => console.error("Error:", err));
