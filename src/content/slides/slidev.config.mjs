import mdAttrs from 'markdown-it-attrs'

/** @type {import('@slidev/types').SlidevConfig} */
const config = {
  theme: 'seriph',
  markdown: {
    markdownItSetup(md) {
      md.use(mdAttrs)
    },
  },
  // 可选：全局样式，避免主题样式干扰 img 宽度
  // css: 'src/styles/slidev.css',
}

export default config
