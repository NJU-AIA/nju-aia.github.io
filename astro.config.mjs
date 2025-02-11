// @ts-check
import { defineConfig } from 'astro/config';
import vue from '@astrojs/vue';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
// import { rehypeHeadingIds } from '@astrojs/markdown-remark';
// import remarkHeaderId from '@astrojs/markdown-remark-header-ids';
import remarkCustomHeaderId from 'remark-custom-header-id';
import tailwindcss from '@tailwindcss/vite';
import react from '@astrojs/react';
import icon from "astro-icon";

// https://astro.build/config
export default defineConfig({
  site: 'https://nju-aia.github.io/',
  integrations: [vue(), mdx(), react(), icon()],

  markdown: {
    remarkPlugins: [remarkCustomHeaderId,remarkMath],
    rehypePlugins: [rehypeKatex]
  },

  vite: {
    plugins: [tailwindcss()]
  }
});