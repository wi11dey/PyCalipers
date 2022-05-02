#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import re

SCALE_REGEX = re.compile(r"^(?P<units>\d*\.?\d+)(?P<unit_name>[^\d./]*)/(?P<pixels>\d*\.?\d+)$")

def multi_gaussian(x, *p):
    y = np.zeros_like(x)
    for i in range(0, len(p), 3):
        mu    = p[i]
        sigma = p[i + 1]
        A     = p[i + 2]
        y += A*np.exp(-0.5*((x - mu)/sigma)**2)
    return y

def multi_gaussian_fit(x):
    from scipy.signal import find_peaks, peak_widths
    from scipy.optimize import curve_fit
    _, properties = find_peaks(x, prominence=2)
    peaks, properties = find_peaks(x, prominence=2, height=0.15*x[properties["left_bases"][0]:properties["right_bases"][-1]].max()) # used to be height=0.15*x[1:-1].max()
    widths, _, _, _ = peak_widths(x, peaks)
    p0 = [param for params in zip(peaks.astype(float), widths, x[peaks]) for param in params]
    bounds = ([], [])
    for i in range(0, len(p0), 3):
        # Mean bounds
        previous_mean = p0[i - 3] if i - 3 >= 0 else 1
        next_mean     = p0[i + 3] if i + 3 < len(p0) else float(len(x)) - 1
        bounds[0].append(previous_mean)
        bounds[1].append(next_mean)

        # Standard deviation bounds
        bounds[0].append(0)
        # bounds[1].append(10)
        bounds[1].append(0.01*len(x))

        # Amplitude bounds
        bounds[0].append(0)
        bounds[1].append(255)
    try:
        p, _ = curve_fit(multi_gaussian, np.arange(len(x), dtype=float), x, p0=p0, bounds=bounds)
        return p
    except BaseException as e:
        raise StopIteration(p0) from e

def derivative(gray):
    # TODO possibly keep stddev of flattening the columns here, then feed into sigma parameter of curve_fit
    return np.abs(np.diff(np.average(gray, axis=0)))

def detect_layers(imagefile=None, margin_top=0, margin_bottom=0, margin_left=0, margin_right=0, scale="1/1", interactive=False):
    import cv2
    from sys import stderr

    print("Reading...", file=stderr)
    img = cv2.imread(imagefile)

    print("Cropping...", file=stderr)
    img = img[slice(margin_top     if margin_top    > 0 else None,
                    -margin_bottom if margin_bottom > 0 else None),
              slice(margin_left    if margin_left   > 0 else None,
                    -margin_right  if margin_right  > 0 else None)]

    print("Detecting orientation...", file=stderr)
    _, thresholded = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresholded = 255 - thresholded
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            best_contour = contour
    center, size, angle = cv2.minAreaRect(best_contour)
    if angle > 60:
        angle -= 90

    print("Rotating by %f°..." % angle, file=stderr)
    height, width = img.shape[:2]
    pivot = (width/2, height/2)
    R = cv2.getRotationMatrix2D(pivot, angle, 1)
    cosine = abs(R[0, 0])
    sine   = abs(R[0, 1])
    rotated_width  = int((height*sine)   + (width*cosine))
    rotated_height = int((height*cosine) + (width*sine))
    rotated_size = (rotated_width, rotated_height)
    R[0, 2] += rotated_width/2  - pivot[0]
    R[1, 2] += rotated_height/2 - pivot[1]
    rotated = cv2.warpAffine(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA), R, rotated_size)

    print("Differentiating...", file=stderr)
    dV = derivative(cv2.cvtColor(rotated[:, :, :3], cv2.COLOR_BGR2GRAY))

    exception = None
    try:
        print("Fitting Gaussian distributions...", file=stderr)
        gaussians = multi_gaussian_fit(dV)

        print("Scaling...", file=stderr)
        scale_match = SCALE_REGEX.match(scale)
        unit_name = scale_match.group("unit_name")
        scale_factor = float(scale_match.group("units"))/float(scale_match.group("pixels"))
        layers = []
        for i in range(3, len(gaussians), 3):
            layers.append(((gaussians[i] - gaussians[i - 3])*scale_factor, math.sqrt(gaussians[i + 1]**2 + gaussians[i - 2]**2)*scale_factor, unit_name))
    except StopIteration as e:
        print("Fitting failed, outputting current state.", file=stderr)
        gaussians = e.value
        exception = e.__cause__

    if interactive:
        from matplotlib import pyplot as plt
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex=ax1)
        ax1.imshow(rotated)
        ax1.set_aspect("auto")
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.plot(dV)
        if exception is None:
            x = np.linspace(0, len(dV), 2000, dtype=float)
            ax2.plot(x, multi_gaussian(x, *gaussians), "r--", linewidth=1)
        means = [gaussians[i] for i in range(0, len(gaussians), 3)]
        ax2.plot(means, # multi_gaussian(means, *gaussians) if exception is None else 
                 dV[[int(mean) for mean in means]], "x")
        plt.tight_layout()
        plt.get_current_fig_manager().set_window_title(imagefile)
        plt.show()
        if exception is not None:
            raise exception

    return layers

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Detect layer thicknesses in an image.")
    parser.add_argument("imagefile", help="Image in which to detect layers.")
    parser.add_argument("--margin-top",    type=int, default=0, help="Number of pixels at the top of the image to ignore.")
    parser.add_argument("--margin-bottom", type=int, default=0, help="Number of pixels at the bottom of the image to ignore.")
    parser.add_argument("--margin-left",   type=int, default=0, help="Number of pixels at the left of the image to ignore.")
    parser.add_argument("--margin-right",  type=int, default=0, help="Number of pixels at the right of the image to ignore.")
    parser.add_argument("--scale", default="1/1", help="<units>[unit name]/<pixels>")
    parser.add_argument("-o", "--output", type=argparse.FileType("w"), action="append", help="Output file in which to store results. Format is automatically selected based on extension. May be specified multilple times.")
    args = parser.parse_args()
    keyword_args = vars(args)
    outputs = keyword_args.pop("output")
    if outputs is None:
        from sys import stdout
        outputs = [stdout]

    layers = detect_layers(**keyword_args, interactive=True)

    if layers is not None:
        import os
        for output in outputs:
            _, ext = os.path.splitext(output.name)
            ext = ext.lower()
            if ext in ["", ".txt"]:
                for i, layer in enumerate(layers):
                    print("Layer %3d:  %7.3f%s ± %5.3f%s" % (i + 1, layer[0], layer[2], layer[1], layer[2]), file=output)
            elif ext in [".csv"]:
                print("Layer #,Thickness,Standard deviation,Units", file=output)
                for i, layer in enumerate(layers):
                    print("%d,%f,%f,%s" % (i + 1, *layer), file=output)
            elif ext in [".tsv"]:
                print("Layer #\tThickness\tStandard deviation\tUnits", file=output)
                for i, layer in enumerate(layers):
                    print("%d\t%f\t%f\t%s" % (i + 1, *layer), file=output)
            output.close()
