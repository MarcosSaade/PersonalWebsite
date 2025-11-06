import React from 'react';
import PageTemplate from '../components/PageTemplate';

import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

import orthoImage from '../images/orthopedics.jpeg';
import uiScreenshot from '../images/orthopedics/ui.png';
import reportPreview from '../images/orthopedics/report-preview.png';

export default function OrthoFlexPage() {
  return (
    <PageTemplate title="Orto-Flex Scanner: Clinical Image Processing for Orthopedics" image={orthoImage}>
      <p>
        In this freelance project, I built a production-grade desktop application for a local orthopedics clinic.
        The goal was to digitize and streamline their patient evaluation process using a hardware-integrated scanner,
        computer vision techniques, and automated PDF report generation. This app is now used in real-world clinical workflows
        across two separate locations, serving approximately 4000 patients per year across two clinics, and has significantly reduced manual work
        in designing orthotic devices.
      </p>

      <h2>The Problem: Manual Process Bottlenecks</h2>
      <p>
        Before this software solution, the clinic's workflow was inefficient and unprofessional:
      </p>
      <ul>
        <li>Clinicians used generic scanner software with no medical specialization</li>
        <li>Patient foot scans included distracting backgrounds and poor contrast</li>
        <li>All measurements and reference lines were drawn manually on printouts</li>
        <li>Staff spent valuable time transcribing patient information between systems</li>
        <li>Reports looked inconsistent and unprofessional, presenting a poor image to patients and referral partners</li>
        <li>During peak hours, patients would experience extended wait times due to these bottlenecks</li>
      </ul>

      <h2>What the Orto-Flex Scanner Does</h2>
      <p>
        The application creates a specialized medical imaging workflow that allows clinicians to scan or upload foot images from patients. 
        These images go through a sophisticated processing pipeline that:
      </p>
      <ul>
        <li>Removes the background and isolates the foot region using contour detection</li>
        <li>Applies skin-tone filtering and visual enhancements to normalize appearance across patients</li>
        <li>Adds anatomical guides and reference lines automatically based on computed foot geometry</li>
        <li>Generates diagnostic heatmaps to highlight pressure points</li>
        <li>Creates a professional, clinic-branded PDF report combining all analysis results and patient data</li>
        <li>Stores historical data in an organized manner for future reference and follow-up visits</li>
      </ul>

      <img src={uiScreenshot} alt="OrthoFlex Scanner UI" className="page-image" />

      <h2>Computer Vision Pipeline</h2>

      <p>
        The heart of the application lies in a custom-built image processing pipeline developed using OpenCV. 
        This pipeline transforms raw scanner input into clean, interpretable visuals that support clinical decision-making. 
        Let's walk through the stages that make this possible.
      </p>

      <h3>Image Acquisition and Preprocessing</h3>
      <p>
        Each scan begins with high-resolution image acquisition through the WIA interface, configured at 300 DPI for clarity. 
        Ensuring consistent lighting and scanner settings across sessions is important for further processing.
        Once captured, the image is checked for orientation and flipped if necessary. This ensures all left and right foot scans share 
        the same coordinate frame.
      </p>

      <h3>Background Removal</h3>
      <p>
        Segmenting the foot from the background is the first major step. I used the HSV (Hue, Saturation, Value) color space 
        instead of RGB because it separates chromatic content (hue) from illumination (value), making it more robust to lighting variations.
        Skin tones typically fall within a hue range of 0–50, with moderate saturation and brightness. These values form the basis of the initial mask.
      </p>

      <p>
        After generating the initial binary mask via HSV thresholding, it must be refined to ensure clean segmentation. 
        Raw masks often include jagged edges, holes, and small disconnected components due to lighting artifacts or sensor noise.
        To address this, I applied several image processing techniques: <i>Gaussian blurring</i>, <i>morphological closing</i>,
        and contour extraction.
      </p>

      <h4>Gaussian Blur</h4>
      <p>
        A Gaussian blur is a smoothing operation that reduces local noise and softens sharp transitions in pixel intensity. 
        It works by replacing each pixel with a weighted average of its neighbors, where weights follow a Gaussian distribution:
      </p>

      <BlockMath math="\text{G}(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}" />

      <p>
        In practice, this is implemented using a kernel, such as <code>(7, 7)</code>, where each pixel is replaced by 
        a weighted average over a 7×7 window. The parameter <code>σ</code> (sigma) controls the extent of the blur. 
        In this case, I used <code>σ = 0</code>, which lets OpenCV choose it automatically based on kernel size.
      </p>

      <h4>Morphological Closing</h4>
      <p>
        Once smoothed, the mask may still contain small gaps or speckled holes. To fix these, I applied a morphological closing operation,
        defined as a dilation followed by erosion. This operation is especially effective at closing narrow holes inside foreground objects.
      </p>

      <p>
        Formally, if <InlineMath math="A"/> is the binary image and <InlineMath math="B"/> is the structuring element (kernel), 
        closing is defined as:
      </p>

      <BlockMath math="A \bullet B = (A \oplus B) \ominus B" />

      <p>
        where <InlineMath math="\oplus"/> denotes dilation and <InlineMath math="\ominus"/> denotes erosion. 
        I used an elliptical kernel of size <code>(5, 5)</code>, which conforms well to rounded anatomical shapes like the foot.
      </p>

      <pre><code className="language-python">{`
      mask = cv2.inRange(hsv, lower_skin, upper_skin)
      mask = cv2.GaussianBlur(mask, (7, 7), 0)
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
      mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=35)
      contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      `}</code></pre>

      <h4>Contour Detection</h4>
      <p>
        After cleaning, I extracted the contours of the binary shape. Contours are curves joining all the continuous points 
        (along a boundary) having the same intensity. OpenCV's <code>findContours</code> function returns a list of such curves.
        The largest one is selected, assuming it corresponds to the foot.
      </p>

      <h4>Convex Hull</h4>
      <p>
        The extracted contour may still have small protrusions, indentations, or noise-induced irregularities. 
        To simplify and regularize its shape, its convex hull is computed.
        Intuitively, the convex hull is the tightest convex shape that encloses all the contour points—
        like stretching a rubber band around the foot shape.
      </p>

      <p>
        Mathematically, the convex hull <InlineMath math="H"/> of a set of points <InlineMath math="S"/> is the smallest convex polygon 
        such that every point in <InlineMath math="S"/> is either inside or on the boundary of <InlineMath math="H"/>.
      </p>

      <p>
        In 2D, this can be computed using algorithms like Graham’s scan or Andrew’s monotone chain (used internally by OpenCV). 
        The result is a simplified and smoothed version of the foot shape, more suitable for visual overlays and 
        subsequent feature extraction.
      </p>

      <p>
        Using the convex hull also removes accidental inward dents in the mask, ensuring the shape remains biologically plausible.
        This is especially important for downstream measurements, like foot width and axis alignment.
      </p>


      <h3>Anatomical Feature Extraction</h3>
      <p>
        Once the foot has been isolated, the app extracts key anatomical landmarks. The heel center is estimated as the lowest point 
        in the contour, while the toe tip is the topmost point along the foot’s medial axis. 
        For measuring width, the app scns horizontal cross-sections from toe to heel, recording the widest distance between contour edges. 
        This allows us to compute the metatarsal width, which is a clinically relevant parameter when designing orthotics.
      </p>

      <pre><code className="language-python">{`
      y_values = np.arange(int(thumb_y), heel_y, 5)
      for y in y_values:
          points = contour[np.abs(contour[:, 1] - y) <= tolerance]
          width = points[:, 0].max() - points[:, 0].min()
          if width > max_width:
              max_width = width
              best_y = y
      `}</code></pre>


      <h2>Hardware Integration Details</h2>
      <p>
        The application uses a hardware abstraction layer that allows it to work with different scanner models,
        though it's optimized for the Epson Perfection series flatbed scanners used by the clinic.
      </p>

      <p>
        In production mode, the app connects directly to the scanner using the Windows Imaging Acquisition (WIA) interface
        through Python's <code>wia</code> module. The integration includes:
      </p>

      <ul>
        <li>Automatic device discovery and connection management</li>
        <li>Custom scanner configuration profiles (resolution, color depth, scan area)</li>
        <li>An event-driven scanning workflow that notifies the application when scanning is complete</li>
        <li>An optional autoclicker system that uses <code>pyautogui</code> to automate dialog interactions</li>
        <li>Fallback mechanisms that allow file imports when hardware is unavailable</li>
      </ul>


      <p>
        The app also includes a configuration panel allowing that allows staff to set up their preferred settings.
      </p>

      <h2>Professional PDF Report Generation</h2>
      <p>
        Once both foot images are processed, the clinician can fill out a dialog with patient details, orthotic material, delivery date,
        and optional clinical notes. This form is pre-filled with the most common values (for example, the most used material) to save even more time, but can be customized for each patient.
        The scanning date is automatically assigned as the current date, and I implemented custom logic to calculate the delivery date automatically based on the clinic's operations.
      </p>

      <p>
        The report generation system then:
      </p>

      <ul>
        <li>Compiles all information into a standardized data structure</li>
        <li>Applies the clinic's branding templates (logos, colors, fonts)</li>
        <li>Positions images, measurements, and text using ReportLab's coordinate system</li>
        <li>Includes a different format for reports with no images, as the clinic sometimes needs to generate them</li>
      </ul>

      <img src={reportPreview} alt="PDF Report Preview" className="page-image" />

      <h2>Complete Tech Stack</h2>
      <ul>
        <li><strong>Python</strong> as the core language for all processing logic</li>
        <li><strong>PyQt5</strong> for the responsive, fullscreen GUI with custom styled components</li>
        <li><strong>OpenCV</strong> for sophisticated image filtering, contour analysis, and heatmap generation</li>
        <li><strong>NumPy</strong> for efficient array operations and mathematical transformations</li>
        <li><strong>ReportLab</strong> for PDF generation with pixel-perfect layout control</li>
        <li><strong>Pillow</strong> for additional image processing capabilities</li>
        <li><strong>WIA</strong> module for Windows scanner integration</li>
        <li><strong>PyAutoGUI</strong> for scanner automation</li>
        <li><strong>QSettings</strong> for persistent user preferences and configuration management</li>
      </ul>

      <h2>Business Impact and Value Creation</h2>
      <p>
        The Orto-Flex Scanner has transformed the clinic's operations in several measurable ways:
      </p>
      
      <ul>
        <li><strong>Time Efficiency:</strong> Reduced per-patient processing time from 10 minutes to under 3 minutes</li>
        <li><strong>Increased Throughput:</strong> The clinic can now serve more patients during peak hours</li>
        <li><strong>Professional Image:</strong> Branded, consistent reports have improved the clinic's image</li>
        <li><strong>Reduced Errors:</strong> Automated measurements have improved orthotic design accuracy by eliminating manual errors. Delivery dates are always accurate, as they are automatically calculated</li>
        <li><strong>Staff Satisfaction:</strong> Technicians report higher job satisfaction with reduced repetitive manual work</li>
        <li><strong>Patient Experience:</strong> Shorter wait times and impressive technical reports have improved patient satisfaction</li>
        <li><strong>Data Consistency:</strong> Digital storage has made follow-up visits more effective with easy access to historical scans</li>
      </ul>

      <p>
        The most significant impact has been on the clinic's professional image. Where patients once received basic, messy printouts with
        hand-drawn lines, they now receive medical-grade diagnostic images with professional layout and consistent branding. This
        elevates the perceived value of their services.
      </p>

      <h2>Technical Challenges Overcome</h2>
      <p>
        Building this system presented several interesting technical problems:
      </p>

      <ul>
        <li><strong>Variable Skin Tones:</strong> Creating a universal skin detection algorithm that works reliably across diverse patient populations required careful calibration and testing</li>
        <li><strong>Integration with Legacy Hardware:</strong> Working with scanner drivers designed for consumer use rather than programmatic access required creative workarounds</li>
        <li><strong>Robust Error Handling:</strong> Building a system that gracefully recovers from hardware failures, unexpected inputs, and user errors without losing patient data</li>
        <li><strong>Cross-Computer Consistency:</strong> Ensuring that colors, measurements, and PDF output looked identical across different workstations with varied displays</li>
      </ul>

      <h2>Other Challenges</h2>
      <p>
        Beyond the technical hurdles, there were several logistical and workflow-related challenges that shaped the development process.
      </p>

      <h3>Remote Development vs. On-Site Testing</h3>
      <p>
        Since the scanner was installed directly on the clinic’s floor and physically embedded near a workstation, I couldn’t test with it during development.
        Instead, I implemented the entire scanner integration layer in developer mode, simulating inputs by uploading pre-scanned images.
        Once the core logic was complete, I had to manually deploy the app onto the clinic’s Windows computer—which was literally mounted on a wall—
        and test it there under real operating conditions.
      </p>

      <p>
        This workflow introduced several friction points. The computer vision pipeline worked well on the sample images I had been sent remotely, 
        but once I began testing with actual scans from the clinic’s hardware, I quickly discovered that lighting conditions, scanner calibration, and image quality varied far more than anticipated. 
        I had to iterate on-site, tweaking the segmentation pipeline and adjusting HSV ranges manually until the results were robust in the wild.
      </p>

      <p>
        Additionally, I was developing the app on Linux, but the final deployment environment was Windows. Normally this wouldn't be a problem, 
        but since the scanning interface relied on WIA (Windows Imaging Acquisition)—a Windows-only API—I had no way to test or even import the scanner code from my development machine. 
        This required careful architectural separation, with conditional logic that could gracefully fall back to upload mode when the scanner wasn’t present.
      </p>

      <h3>Client Communication and Iteration</h3>
      <p>
        Another core challenge was maintaining effective communication with the client throughout the project. 
        The application was custom-built for a specific clinical workflow, and the client made frequent revisions and feature requests as the product evolved.
      </p>

      <p>
        I was in near-constant contact with the clinic’s staff, exchanging screenshots of UI components, sharing PDF previews, and 
        iterating on details like font size, report layout, and labeling terminology. While demanding, this process helped me better align the tool 
        with real-world user expectations and improve usability in subtle but impactful ways.
      </p>


      <h2>Reflection and Learnings</h2>
      <p>
        This was my first time building a medical-adjacent desktop tool for real-world clinical use. It challenged me to design a
        user-friendly, robust system that worked seamlessly under pressure. From integrating scanner hardware to dynamically computing
        foot metrics, every component required careful thought and testing. 
      </p>

      <p>
        Working closely with orthopedic specialists taught me the importance of domain-specific knowledge in software design.
        Many features I initially thought would be valuable were replaced by simpler solutions that better matched the clinical workflow.
      </p>

      <p>
        I'm proud that the app is now used in day-to-day operations in two clinics, serving approximately 4000 patients yearly — 
        saving time and improving the quality of orthotic evaluations. The clinic director has reported that the system paid for itself through increased throughput and has become an essential part of their practice.
      </p>

      <p>
        This project deepened my experience with medical image processing, UI/UX design for professional environments, and PDF rendering — 
        all valuable skills I hope to continue applying in healthcare and clinical software development.
      </p>
    </PageTemplate>
  );
}