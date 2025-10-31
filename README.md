# Sammie-Roto 2
**S**egment **A**nything **M**odel with **M**atting **I**ntegrated **E**legantly

![Sammie-Roto 2 screenshot](https://github.com/user-attachments/assets/dfd86bec-f128-47e6-b2e4-dfc9b340236a)

Sammie-Roto 2 is a full-featured, cross-platform desktop application for AI assisted masking of video clips. It has 3 primary functions:
- Video Segmentation using [SAM2](https://github.com/facebookresearch/sam2)
- Video Matting using [MatAnyone](https://github.com/pq-yang/MatAnyone)
- Video Object Removal using [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)

Please add a Github Star if you find it useful!

### Updates
- [10/31/2025] Release of Sammie-Roto 2 Beta.

### Documentation:
[Documentation and usage guide](https://github.com/Zarxrax/Sammie-Roto-2/wiki)

### Installation (Windows):
- Download latest Windows version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive.
- Run 'install_dependencies.bat' and follow the prompt.
- Run 'run_sammie.bat' to launch the software.

Everything is self-contained in the Sammie-Roto folder. If you want to remove the application, simply delete this folder. You can also move the folder.

### Installation (Linux, Mac)
- Ensure [Python](https://www.python.org/) is installed (version 3.10 or higher)
- Download latest Linux/Mac version from [releases](https://github.com/Zarxrax/Sammie-Roto-2/releases)
- Extract the zip archive.
- Open a terminal and navigate to the Sammie-Roto folder that you just extracted from the zip.
- Execute the following command: `bash install_dependencies.sh` then follow the prompt.
- MacOS users: double-click "run_sammie.command" to launch the program. Linux users: `bash run_sammie.command` or execute the file however you prefer.

### Acknowledgements
* [SAM 2](https://github.com/facebookresearch/sam2)
* [MatAnyone](https://github.com/pq-yang/MatAnyone)
* [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (for optimized MatAnyone)
* [MiniMax-Remover](https://github.com/zibojia/MiniMax-Remover)
