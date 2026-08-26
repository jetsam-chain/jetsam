Jetsam Native Release
=======================

Each GitHub release contains two independent product lines.

GUI Wallet packages (ordinary users):

  Linux:   jetsam-gui-vVERSION-linux-ARCH.deb
  Windows: jetsam-gui-vVERSION-windows-x86_64-setup.exe
  macOS:   jetsam-gui-vVERSION-macos-ARCH.dmg

The GUI package exposes only the Jetsam wallet application. Its full node
is bundled as a private application component and is supervised by the wallet.
The user does not need to start a daemon or use a terminal.

Core archives (node operators and miners):

  jetsam        full node and built-in miner
  jetsam-cli    wallet and node command-line client
  jetsam-miner  external proof-of-work miner
  LICENSE/NOTICE  Apache-2.0 distribution terms and project notices

Hardware check
--------------

Before creating node or wallet data, run:

  jetsam --check-hardware

Production requires SSE4.1 and PCLMULQDQ on x86-64, or NEON and PMULL on
ARM64. The executable selects wider AVX2+VPCLMULQDQ or AVX-512 kernels
automatically. Unsupported hardware exits with a readable diagnostic; the
scalar reference backend is not used for production.

Verify the download
-------------------

Download SHA256SUMS from the same GitHub release as this archive:

  https://github.com/ignotusnemo/jetsam/releases

Before extracting or running anything, compute the archive's SHA-256 digest
and compare it with the matching line in SHA256SUMS.

Linux:

  sha256sum <downloaded-archive>

macOS:

  shasum -a 256 <downloaded-archive>

Windows PowerShell:

  Get-FileHash <downloaded-archive> -Algorithm SHA256

Never run an archive whose digest does not match.

GUI Wallet — Linux
------------------

Open the downloaded .deb in the system Software application, or install it
from a terminal:

  sudo apt install ./jetsam-gui-vVERSION-linux-ARCH.deb

Launch Jetsam from the desktop application menu. Removing the package does
not remove wallet data from the user's home directory.

GUI Wallet — Windows
--------------------

Run the downloaded setup.exe and launch Jetsam from the Start menu. The
installer is per-user and does not require administrator privileges by
default.

Until the project uses an Authenticode certificate, Microsoft Defender
SmartScreen may display a warning. After verifying SHA256SUMS, select
"More info" and then "Run anyway".

GUI Wallet — macOS
------------------

Open the downloaded DMG and drag Jetsam.app to Applications. Launch it from
Applications like any other app.

Until the project uses an Apple Developer ID certificate, macOS may block the
first launch. After verifying SHA256SUMS, Control-click Jetsam, choose Open,
and confirm. If necessary, use Privacy & Security -> Open Anyway.

Core archive — Linux
--------------------

Open a terminal in the extracted directory:

  ./jetsam --help
  ./jetsam-cli --help
  ./jetsam-miner --help

Core archive — macOS
--------------------

If Gatekeeper blocks a verified download, remove only the quarantine
attributes from the three extracted binaries:

  xattr -d com.apple.quarantine ./jetsam
  xattr -d com.apple.quarantine ./jetsam-cli
  xattr -d com.apple.quarantine ./jetsam-miner

If xattr reports that an attribute does not exist, no action is required.
Then run:

  ./jetsam --help

Core archive — Windows
----------------------

PowerShell can unblock all three verified extracted executables at once:

  Get-ChildItem .\*.exe | Unblock-File

Then run:

  .\jetsam.exe --help

Node data
---------

The first node start creates its configuration and persistent data under:

  Linux/macOS:  ~/.jetsam/
  Windows:      %USERPROFILE%\.jetsam\

The wallet key is stored in data/wallet.key and is not password-encrypted.
Back it up and protect it before receiving funds.

Documentation: https://docs.jetsam.org/
Source:        https://github.com/ignotusnemo/jetsam
