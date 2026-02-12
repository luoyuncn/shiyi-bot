“我要用 Python 的 Textual 框架复刻 Claude Code CLI 的界面风格。请严格按照以下 ASCII 布局生成代码：

整体布局：全屏单列布局，没有侧边栏，没有 Header。

顶部状态：右上角仅保留一个微小的 'Model: Claude 3.7' 灰色标签（Dock Top）。

中间对话流 (Scrollable Stream)：

占据屏幕 90% 空间。

User 消息：右对齐，深灰色背景泡泡，圆角。

AI 消息：左对齐，背景透明，Markdown 渲染，代码块高亮。

思考状态 (Thinking)：一个折叠的 details 区域，显示 'Thinking...' 动画。

底部输入区 (Floating Input)：

不固定在最底部，而是像一个‘悬浮岛’一样浮在内容上方一点点（或者有明显的 padding）。

输入框没有显眼的边框，只有左侧有一个彩色的 ❯ 提示符。”

“请使用以下 TCSS 样式规则来确保视觉还原：

全局背景: 使用深色背景 #1a1b26 (Night Owl 风格) 或 #000000 (纯黑)。

输入框 (Input): 去掉默认的边框 (border: none;)，背景色设为 #24283b，文字颜色 #c0caf5。

用户消息: background: #2e3c56; color: white; padding: 1 2; border-radius: 1;，且 align: right。

AI 消息: 背景透明，Markdown 中的代码块背景需为黑色 #111。

提示符: 输入框前的 ❯ 符号使用橙色 #ff9e64 或紫色 #bb9af7。”

宽度限制 (width: 80%)：这是“高级感”的秘诀。不要让文字横向铺满整个屏幕，留出左右两侧的空白，阅读体验会瞬间提升一个档次。

隐藏滚动条 (scrollbar-size: 0 0)：Claude Code 看起来很干净是因为你看不到丑陋的滚动条，所有内容都是自然滑动的。

Markdown：Textual 的 Static 组件原生支持 Markdown，一定要用它来渲染 AI 的回复。