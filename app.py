import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Design Change Detector + Green Shadow", layout="centered")
st.title("🎯 Design Change Detector with Green Shadow (Major Changes Only)")

st.markdown("""
Upload **two PNG images** (e.g., CAD versions).  
I’ll highlight **major form changes** with red boxes,  
and show **missing old parts** as green shadows.
""")

img1_file = st.file_uploader("Upload OLD Image (Reference)", type=["png"])
img2_file = st.file_uploader("Upload NEW Image (Modified)", type=["png"])

min_area = st.slider("Minimum major change size (area)", 100, 5000, 200, step=50)

if img1_file and img2_file:
    with st.spinner("🔍 Processing images..."):
        # Load images as RGBA
        img1 = Image.open(img1_file).convert("RGBA")
        img2 = Image.open(img2_file).convert("RGBA")

        # Resize new image to old if sizes mismatch
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)

        # Convert to arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # 1) Detect missing old parts (green shadow)
        alpha1 = arr1[:, :, 3]
        alpha2 = arr2[:, :, 3]
        missing_mask = (alpha1 > 0) & (alpha2 == 0)  # old has alpha, new has none

        # Start result as new image RGB
        result_img = arr2[:, :, :3].copy()

        # Apply green shadow where old part missing
        # Blend green with existing pixels for subtle shadow
        green = np.array([0, 255, 0], dtype=np.uint8)
        # We'll blend 50% green with existing color
        alpha_blend = 0.5
        result_img[missing_mask] = (
            (result_img[missing_mask].astype(float) * (1 - alpha_blend)) + 
            (green.astype(float) * alpha_blend)
        ).astype(np.uint8)

        # 2) Detect major changes (red boxes)
        # Convert images to grayscale (RGB part)
        gray1 = cv2.cvtColor(arr1[:, :, :3], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(arr2[:, :, :3], cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, diff_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        major_change_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # Red box
                cv2.putText(result_img, "Major Change", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                major_change_count += 1

        st.subheader(f"📸 Major Design Differences Detected: {major_change_count}")

        if major_change_count == 0:
            st.info("No major changes detected with this threshold. Try lowering the slider!")

        result_pil = Image.fromarray(result_img)
        st.image(result_pil, use_column_width=True)

        buf = io.BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button("📥 Download Result Image", data=buf.getvalue(), file_name="design_diff.png", mime="image/png")
