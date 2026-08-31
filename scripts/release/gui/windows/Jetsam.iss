#ifndef MyAppVersion
  #error MyAppVersion is required
#endif
#ifndef SourceDir
  #error SourceDir is required
#endif
#ifndef NumericVersion
  #error NumericVersion is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif
#ifndef OutputBaseFilename
  #error OutputBaseFilename is required
#endif
#ifndef IconFile
  #error IconFile is required
#endif
#ifndef LicenseFile
  #error LicenseFile is required
#endif
#ifndef NoticeFile
  #error NoticeFile is required
#endif

[Setup]
AppId={{C36DA503-269C-4BAA-ACD9-D3EC3632F445}
AppName=Jetsam
AppVersion={#MyAppVersion}
AppVerName=Jetsam {#MyAppVersion}
AppPublisher=Jetsam
AppPublisherURL=https://jetsamchain.com/
AppSupportURL=https://github.com/ignotusnemo/jetsam/issues
AppUpdatesURL=https://github.com/ignotusnemo/jetsam/releases
DefaultDirName={localappdata}\Programs\Jetsam
DefaultGroupName=Jetsam
DisableProgramGroupPage=yes
AllowNoIcons=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename={#OutputBaseFilename}
SetupIconFile={#IconFile}
LicenseFile={#LicenseFile}
UninstallDisplayIcon={app}\Jetsam.exe
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
SetupLogging=yes
VersionInfoVersion={#NumericVersion}
VersionInfoCompany=Jetsam
VersionInfoDescription=Jetsam Wallet Installer
VersionInfoProductName=Jetsam
VersionInfoProductVersion={#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "chinesesimplified"; MessagesFile: "{#SourcePath}\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\jetsam-gui.exe"; DestDir: "{app}"; DestName: "Jetsam.exe"; Flags: ignoreversion
Source: "{#SourceDir}\jetsam.exe"; DestDir: "{app}"; DestName: "jetsam-node.exe"; Flags: ignoreversion
Source: "{#LicenseFile}"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "{#NoticeFile}"; DestDir: "{app}"; DestName: "NOTICE.txt"; Flags: ignoreversion

[Icons]
Name: "{group}\Jetsam"; Filename: "{app}\Jetsam.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Jetsam"; Filename: "{app}\Jetsam.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Jetsam.exe"; Description: "{cm:LaunchProgram,Jetsam}"; Flags: nowait postinstall skipifsilent
