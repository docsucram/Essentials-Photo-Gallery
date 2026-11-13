; Script generated for Inno Setup 6
; Essentials Photo Gallery (PhotoGallery64) Windows x64 Installer

#define MyAppName "Essentials Photo Gallery"
#define MyAppVersion "0.12.0"
#define MyAppPublisher "Essentials Software"
#define MyAppURL "https://github.com/docsucram/Essentials-Photo-Gallery"
#define MyAppExeName "EssentialsPhotoGallery.exe"

[Setup]
; App identification
AppId={{9F84D42B-35EA-4682-A62E-2A4BF10B39F1}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Architecture: 64-bit Windows only
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=dist_installer
OutputBaseFilename=EssentialsPhotoGallery_Setup_x64
SetupIconFile=resources\icons\essentials_photo_gallery.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "fileassoc"; Description: "Associate Essentials Photo Gallery with common image files (.jpg, .jpeg, .png, .webp)"; GroupDescription: "File Associations:"

[InstallDelete]
; Clean out previous PyInstaller runtime and assets before installing to prevent stale packages/DLLs
Type: filesandordirs; Name: "{app}\_internal"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\resources"

[Files]
; Main application binaries and dependencies from PyInstaller dist
Source: "dist\EssentialsPhotoGallery\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Application icon
Source: "resources\icons\essentials_photo_gallery.ico"; DestDir: "{app}\resources\icons"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\resources\icons\essentials_photo_gallery.ico"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\resources\icons\essentials_photo_gallery.ico"; Tasks: desktopicon

[Registry]
; App Paths registration for Windows Run dialog (Win+R -> EssentialsPhotoGallery.exe)
Root: HKA; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExeName}"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExeName}"; ValueType: string; ValueName: "Path"; ValueData: "{app}"; Flags: uninsdeletekey

; "Open with Essentials Photo Gallery" context menu for general image file classes
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\image\shell\OpenWithEssentialsPhotoGallery"; ValueType: string; ValueName: ""; ValueData: "Open with Essentials Photo Gallery"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\image\shell\OpenWithEssentialsPhotoGallery"; ValueType: string; ValueName: "Icon"; ValueData: """{app}\resources\icons\essentials_photo_gallery.ico"""; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\image\shell\OpenWithEssentialsPhotoGallery\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Flags: uninsdeletekey; Tasks: fileassoc

; ProgID registration
Root: HKA; Subkey: "Software\Classes\EssentialsPhotoGallery.AssocFile"; ValueType: string; ValueName: ""; ValueData: "Image File"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\EssentialsPhotoGallery.AssocFile\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\resources\icons\essentials_photo_gallery.ico,0"; Flags: uninsdeletekey; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\EssentialsPhotoGallery.AssocFile\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Flags: uninsdeletekey; Tasks: fileassoc

; OpenWithProgids registration for specific image extensions
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.jpg\OpenWithProgids"; ValueType: string; ValueName: "EssentialsPhotoGallery.AssocFile"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.jpeg\OpenWithProgids"; ValueType: string; ValueName: "EssentialsPhotoGallery.AssocFile"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.png\OpenWithProgids"; ValueType: string; ValueName: "EssentialsPhotoGallery.AssocFile"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.webp\OpenWithProgids"; ValueType: string; ValueName: "EssentialsPhotoGallery.AssocFile"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.bmp\OpenWithProgids"; ValueType: string; ValueName: "EssentialsPhotoGallery.AssocFile"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.heic\OpenWithProgids"; ValueType: string; ValueName: "EssentialsPhotoGallery.AssocFile"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.tif\OpenWithProgids"; ValueType: string; ValueName: "EssentialsPhotoGallery.AssocFile"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc
Root: HKA; Subkey: "Software\Classes\SystemFileAssociations\.tiff\OpenWithProgids"; ValueType: string; ValueName: "EssentialsPhotoGallery.AssocFile"; ValueData: ""; Flags: uninsdeletevalue; Tasks: fileassoc

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
