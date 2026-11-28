; Inno Setup Installer Script for ModelForge AI (Windows x64)
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

[Setup]
AppId={{8F9E5A14-4C93-4D2B-9A3E-71B301F0D921}
AppName=ModelForge AI
AppVersion={#MyAppVersion}
AppPublisher=ModelForge AI
DefaultDirName={autopf}\ModelForge AI
DefaultGroupName=ModelForge AI
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
OutputDir=..\..\release_artifacts
OutputBaseFilename=ModelForge-Windows-x64-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\..\release_artifacts\win_bundle\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\ModelForge AI"; Filename: "{app}\modelforge_app.exe"
Name: "{group}\Uninstall ModelForge AI"; Filename: "{uninstallexe}"
Name: "{autodesktop}\ModelForge AI"; Filename: "{app}\modelforge_app.exe"
