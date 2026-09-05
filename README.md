# Sammie-Roto 2
**S**egment **A**nything **M**odel with **M**atting **I**ntegrated **E**legantly

![Sammie-Roto 2 screenshot](https://github.com/user-attachments/assets/bc2c99c8-4039-49f1-94ed-65f104a83e8d)

[![GitHub Downloads](https://img.shields.io/github/downloads/Zarxrax/Sammie-Roto-2/total)](https://github.com/Zarxrax/Sammie-Roto-2/releases)
[![GitHub Code License](https://img.shields.io/github/license/Zarxrax/Sammie-Roto-2)](LICENSE)
[![Discord](https://img.shields.io/discord/1437589475369811970?label=Discord&color=blue)](https://discord.gg/jb5qrFyGFF)

**Please add a Github Star if you find it useful!**

Sammie-Roto 2 is an easy-to-use, cross-platform desktop application for AI assisted masking of video clips. It has 3 primary functions:
- Video Segmentation using [SAM2](https://github.com/facebookresearch/sam2)
- Video Matting using [MatAnyone](https://github.com/pq-yang/MatAnyone), [MatAnyone 2](https://github.com/pq-yang/MatAnyone2), and [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
- Video Object Removal using [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)

Sammie-Roto 2 is free and open source, but runs models produced by several external projects and organizations. Some models may have restrictions on commercial usage. Please check with the relevant model provider if you have questions regarding licensing.

### Documentation and Tutorials:
## [Wiki Documentation and usage guide](https://github.com/Zarxrax/Sammie-Roto-2/wiki)

[![Quick Start Video](https://img.youtube.com/vi/m0iZpxsZJcE/0.jpg)](https://www.youtube.com/watch?v=m0iZpxsZJcE)

*This video does not cover new features and changes added since version 2.0

## Need more help? [Join the Discord](https://discord.gg/jb5qrFyGFF)!

### Installation (Windows):
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive to any location that doesn't restrict write permissions (don't put it in Program Files)
- Run 'install.bat' and follow the prompt.
- Run 'run_sammie.bat' to launch the software (or double click the desktop shortcut).

Everything is self-contained in the Sammie-Roto folder. If you want to remove the application, simply delete this folder.

### Installation (Linux, Mac)
- Download latest version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive.
- Open a terminal and navigate to the Sammie-Roto folder that you just extracted from the zip.
- Execute the following command in the terminal: `bash install.sh` then follow the prompt.
- MacOS users: double-click the desktop icon to launch the program. Linux users: `bash run_sammie.sh` or find it in the applications menu.

### Updates
**Full Changelog can be seen under [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)**
- [09/05/2026] 2.4.1 - Premilinary support for AMD GPUs on Windows, some bugfixes.
- [08/22/2026] 2.4.0 - New installer/updater, additional segmentation tracking options, many small fixes and improvements.
- [04/17/2026] 2.3.3 - Several large performance optimizations, and colorspace conversions are now handled correctly.
- [04/10/2026] 2.3.2 - Improved temporal stability for VideoMaMa.
- [04/02/2026] 2.3.1 - Added a live preview during segmentation while holding the shift key.
- [03/27/2026] 2.3.0 - Added VideoMaMa model, added option to combine objects when matting, fixed major segmentation bug, and more.
- [03/08/2026] 2.2.0 - Added MatAnyone2 model.
- [01/18/2026] 2.1.1 - Rebuilt the export dialog, slightly faster application startup, bug fixes.
- [12/16/2025] 2.1.0 - Added In/Out markers. Modifying points no longer deletes tracking data. Enabled half-precision for much faster segmentation. Added EfficientTAM model.
- [11/23/2025] 2.0.0 - First stable release. Includes several new features and bugfixes. New quick-start video tutorial and [Discord server](https://discord.gg/jb5qrFyGFF).
- [10/31/2025] Release of Sammie-Roto 2 Beta.

### Acknowledgements
* [SAM 2](https://github.com/facebookresearch/sam2)
* [EfficientTAM](https://github.com/yformer/EfficientTAM)
* [MatAnyone](https://github.com/pq-yang/MatAnyone) & [MatAnyone2](https://github.com/pq-yang/MatAnyone2)
* [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (for optimized MatAnyone code)
* [VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa)
* [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)
* Some icons by [Yusuke Kamiyamane](http://p.yusukekamiyamane.com/)
