# ShiYi TUI 字体指南

ShiYi TUI 默认使用 Unicode emoji / ASCII 符号，**无需安装额外字体**即可正常使用。

如果你希望获得更精致的图标体验（工具图标、状态标识等），可以安装一款 **Nerd Font** 并在配置中启用。

## 什么是 Nerd Font？

[Nerd Fonts](https://www.nerdfonts.com/) 是一组在常用编程字体基础上补丁了 3600+ 图标字形的字体家族，广泛用于终端美化（Oh My Zsh、Starship、Neovim 等）。

## 推荐字体

| 字体 | 风格 | 说明 |
|------|------|------|
| **JetBrainsMono Nerd Font** | 现代等宽 | JetBrains 出品，连字支持，极佳可读性 |
| **FiraCode Nerd Font** | 编程连字 | 经典编程字体 + Nerd Font 图标 |
| **CascadiaCode Nerd Font** | 微软风格 | Windows Terminal 默认字体的 Nerd 版本 |
| **Monaspace Krypton** | 赛博朋克 | GitHub 出品，机械工业风（需另行补丁） |
| **Hack Nerd Font** | 简洁清晰 | 专为终端设计的等宽字体 |

## 安装方法

### Windows

1. 访问 https://www.nerdfonts.com/font-downloads
2. 下载你喜欢的字体 zip（推荐 JetBrainsMono）
3. 解压后全选 `.ttf` 文件 → 右键 → **为所有用户安装**
4. 在终端设置中选择对应字体：
   - **Windows Terminal**: 设置 → 配置文件 → 外观 → 字体 → 选择 `JetBrainsMono Nerd Font`
   - **VSCode 终端**: 设置 → `terminal.integrated.fontFamily` → `'JetBrainsMono Nerd Font'`

### macOS

```bash
brew install --cask font-jetbrains-mono-nerd-font
```

然后在终端 app（iTerm2 / Terminal.app）的偏好设置中选择该字体。

### Linux

```bash
# Arch / Manjaro
sudo pacman -S ttf-jetbrains-mono-nerd

# Ubuntu / Debian
mkdir -p ~/.local/share/fonts
cd ~/.local/share/fonts
curl -fLo "JetBrainsMono.zip" https://github.com/ryanoasis/nerd-fonts/releases/latest/download/JetBrainsMono.zip
unzip JetBrainsMono.zip -d JetBrainsMono
fc-cache -fv
```

## 启用 Nerd Font 模式

安装字体并配置终端后，编辑 `config/config.yaml`：

```yaml
tui:
  nerd_font: true
```

重新启动 ShiYi 即可看到 Nerd Font 图标。

## 验证是否生效

启动后欢迎页面会显示：
- `nerd_font: false` → 使用 emoji/ASCII 图标（🍊 ❯ ▸ 等）
- `nerd_font: true` → 使用 Nerd Font 图标（󰕄 󰕌 󰁔 等）

如果启用后看到方块 `□` 或问号，说明终端未正确加载 Nerd Font，请检查终端字体设置。
