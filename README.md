# Sammie-Roto 2
**S**egment **A**nything **M**odel with **M**atting **I**ntegrated **E**legantly

![Sammie-Roto 2 screenshot](https://github.com/user-attachments/assets/bc2c99c8-4039-49f1-94ed-65f104a83e8d)

[![GitHub Downloads](https://img.shields.io/github/downloads/Zarxrax/Sammie-Roto-2/total)](https://github.com/Zarxrax/Sammie-Roto-2/releases)
[![GitHub Code License](https://img.shields.io/github/license/Zarxrax/Sammie-Roto-2)](LICENSE)
[![Discord](https://img.shields.io/discord/1437589475369811970?label=Discord&color=blue)](https://discord.gg/jb5qrFyGFF)

Sammie-Roto 2 is a full-featured, cross-platform desktop application for AI assisted masking of video clips. It has 3 primary functions:
- Video Segmentation using [SAM2](https://github.com/facebookresearch/sam2)
- Video Matting using [MatAnyone](https://github.com/pq-yang/MatAnyone)
- Video Object Removal using [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)

**Please add a Github Star if you find it useful!**

### Updates
**Full Changelog can be seen under [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)**
- [12/16/2025] 2.1.0 - Added In/Out markers. Modifying points no longer deletes tracking data. Enabled half-precision for much faster segmentation. Added EfficientTAM model.
- [11/23/2025] 2.0.0 - First stable release. Includes several new features and bugfixes. New quick-start video tutorial and [Discord server](https://discord.gg/jb5qrFyGFF).
- [10/31/2025] Release of Sammie-Roto 2 Beta.

### Documentation and Tutorials:
[Documentation and usage guide](https://github.com/Zarxrax/Sammie-Roto-2/wiki)

[![Quick Start Video](https://img.youtube.com/vi/m0iZpxsZJcE/0.jpg)](https://www.youtube.com/watch?v=m0iZpxsZJcE)

### Installation (Windows):
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive to any location that doesn't restrict write permissions (so not in Program Files)
- Run 'install_dependencies.bat' and follow the prompt.
- Run 'run_sammie.bat' to launch the software.

Everything is self-contained in the Sammie-Roto folder. If you want to remove the application, simply delete this folder. You can also move the folder.

### Installation (Linux, Mac)
- MacOS users: Make sure Homebrew is installed.
- Ensure [Python](https://www.python.org/) is installed (version 3.10 or higher, 3.12 recommended)
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive.
- Open a terminal and navigate to the Sammie-Roto folder that you just extracted from the zip.
- Execute the following command: `bash install_dependencies.sh` then follow the prompt.
- MacOS users: double-click "run_sammie.command" to launch the program. Linux users: `bash run_sammie.command` or execute the file however you prefer.

### Acknowledgements
* [SAM 2](https://github.com/facebookresearch/sam2)
* [EfficientTAM](https://github.com/yformer/EfficientTAM)
* [MatAnyone](https://github.com/pq-yang/MatAnyone)
* [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)
* [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (for optimized MatAnyone)
* Some icons by [Yusuke Kamiyamane](http://p.yusukekamiyamane.com/)
