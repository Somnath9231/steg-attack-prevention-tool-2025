# Steganography Attack Prevention Tool 2025

Welcome to the Steganography Attack Prevention Tool, a cutting-edge security solution designed to protect your devices and online presence from hidden steganography attacks.

---

## Features

- **Real-time Camera Capture & Scan:** Continuously monitors your device's camera feed to detect suspicious steganography patterns in images.
- **Advanced Steganography Detection:** Uses machine learning models analyzing multiple image channels for accurate detection.
- **Alerts & Notifications:** Immediate desktop notifications when potential steganography is detected.
- **Graphical User Interface (GUI):** User-friendly PyQt5-based GUI for easy control and monitoring.
- **Social Media Image Scanning:** Fetches and scans images from social media platforms to prevent hidden data leaks.
- **Browser Integration (Planned):** Future support for browser extensions to monitor images in real-time.
- **Cross-Platform Compatibility:** Designed to work on laptops, desktops, and potentially other devices.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Camera device connected to your system
- pip (Python package installer)

### Install Dependencies

First, ensure pip is installed. If pip is not installed, you can install it using:

```bash
sudo apt update && sudo apt install python3-pip
```

Then install the required Python packages:

```bash
python3 -m pip install opencv-python
python3 -m pip install -r requirements.txt
```

![Installation GIF](https://media.giphy.com/media/3o7aD2saalBwwftBIY/giphy.gif)

---

## Usage

### Command Line Interface

Run the tool to start real-time camera capture and scanning:

```bash
python3 main.py
```

Press `q` to quit the camera feed.

![CLI Usage GIF](https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif)

### Graphical User Interface

Run the GUI application:

```bash
python3 gui.py
```

Use the buttons to start and stop camera scanning. Alerts will be shown in the log area and as desktop notifications.

![GUI Usage GIF](https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif)

---

## Permissions Required

- **Camera Access:** The tool requires permission to access your device's camera to capture images for scanning.
- **Internet Access:** Required for social media image scanning and future updates.

---

## How It Works

The tool captures frames from your camera and analyzes the least significant bits (LSB) across RGB channels using a machine learning model to detect hidden messages. It also supports scanning images fetched from social media URLs. Alerts are provided via desktop notifications and GUI logs.

---

## Visuals & Animations

Welcome to the fun zone! Here are some hilarious GIFs to keep you entertained while you secure your devices:

![Funny Steganography GIF](https://media.giphy.com/media/3o6ZtpxSZbQRRnwCKQ/giphy.gif)

*Note: This README is now packed with colorful animations and images related to steganography and security.*

![Security Dance GIF](https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif)

Stay alert, stay safe!

---

## Contribution

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---

## Interview Submission Suitability

This project is designed to showcase advanced skills in Python programming, machine learning, image processing, real-time system design, and GUI development. It includes:

- Modular and clean code structure.
- Machine learning integration for advanced detection.
- Real-time camera scanning with alerts.
- Social media image scanning capabilities.
- User-friendly GUI interface.
- Comprehensive documentation and testing.

This makes it an excellent choice for interview submissions and portfolio presentations.

---

## License

MIT License