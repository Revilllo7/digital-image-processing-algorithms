/**
 * Custom scale script for Photoshop
 * Implements interpolation using (max + min)/2 of the four nearest pixels.
 * Input: open image (RGB or Grayscale)
 * Output: new document at target size (500x650)
 */

// #target photoshop
app.bringToFront();

// ===== CONFIGURATION =====
var targetWidth = 500;
var targetHeight = 650;
// ==========================

// Ensure Photoshop uses pixels
var originalRuler = app.preferences.rulerUnits;
app.preferences.rulerUnits = Units.PIXELS;

// Utility: safely get the brightness value of a pixel in a given channel
function getPixelChannelValue(doc, x, y, channelIndex) {
    // Duplicate small 1x1 region for sampling
    var tmp = doc.duplicate();
    tmp.selection.select([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]]);
    tmp.selection.copy();

    var sample = app.documents.add(1, 1, doc.resolution, "sample", NewDocumentMode.GRAYSCALE);
    sample.paste();
    sample.flatten();

    var hist = sample.activeChannels[0].histogram; // fixed: access channel histogram
    var total = 0;
    for (var i = 0; i < hist.length; i++) total += hist[i] * i;
    var avg = total / 1.0; // average value
    sample.close(SaveOptions.DONOTSAVECHANGES);
    tmp.close(SaveOptions.DONOTSAVECHANGES);
    return avg;
}

// Main scaling
function scaleCustom(doc, wIn, hIn, wOut, hOut, channelIndex) {
    var newDoc = app.documents.add(wOut, hOut, doc.resolution, doc.name + "_scaled_" + channelIndex, NewDocumentMode.GRAYSCALE);
    var srcStepX = wIn / wOut;
    var srcStepY = hIn / hOut;

    for (var yOut = 0; yOut < hOut; yOut++) {
        for (var xOut = 0; xOut < wOut; xOut++) {
            var srcX = xOut * srcStepX;
            var srcY = yOut * srcStepY;

            var x0 = Math.floor(srcX);
            var y0 = Math.floor(srcY);
            var x1 = Math.min(x0 + 1, wIn - 1);
            var y1 = Math.min(y0 + 1, hIn - 1);

            var v1 = getPixelChannelValue(doc, x0, y0, channelIndex);
            var v2 = getPixelChannelValue(doc, x1, y0, channelIndex);
            var v3 = getPixelChannelValue(doc, x0, y1, channelIndex);
            var v4 = getPixelChannelValue(doc, x1, y1, channelIndex);

            var arr = [v1, v2, v3, v4];
            arr.sort(function (a, b) { return a - b; });
            var result = (arr[0] + arr[3]) / 2.0;

            // Draw pixel
            var gray = new SolidColor();
            gray.gray.gray = (result * 100.0) / 255.0; // convert 0–255 → %
            newDoc.selection.select([[xOut, yOut], [xOut + 1, yOut], [xOut + 1, yOut + 1], [xOut, yOut + 1]]);
            newDoc.selection.fill(gray);
        }
    }
    newDoc.flatten();
    return newDoc;
}

// Entry point
if (app.documents.length === 0) {
    alert("Open an image first.");
} else {
    var doc = app.activeDocument;
    if (doc.mode != DocumentMode.RGB) {
        alert("Convert image to RGB 8-bit first (Image > Mode > RGB Color).");
    } else {
        var wIn = doc.width.as("px");
        var hIn = doc.height.as("px");
        var scaledDocs = [];
        var merged = app.documents.add(targetWidth, targetHeight, doc.resolution, "merged_scaled", NewDocumentMode.RGB);

        for (var i = 0; i < 3; i++) {
            scaledDocs.push(scaleCustom(doc, wIn, hIn, targetWidth, targetHeight, i));
            app.activeDocument = scaledDocs[i];
            scaledDocs[i].selection.selectAll();
            scaledDocs[i].selection.copy();
            app.activeDocument = merged;
            merged.activeChannels = [merged.channels[i]];
            merged.paste();
            scaledDocs[i].close(SaveOptions.DONOTSAVECHANGES);
        }

        merged.flatten();
        alert("Scaling complete! Output document: 'merged_scaled'");
    }
}

app.preferences.rulerUnits = originalRuler;
