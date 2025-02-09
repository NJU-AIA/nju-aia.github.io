// @ts-check
import { defineConfig } from 'astro/config';
import vue from '@astrojs/vue';
import mdx from '@astrojs/mdx';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import tailwindcss from '@tailwindcss/vite';
import react from '@astrojs/react';
// https://astro.build/config
export default defineConfig({
  site: 'https://nju-aia.github.io/',
  integrations: [vue(), mdx(), react()],

  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex]
  },

  vite: {
    plugins: [tailwindcss()]
  }
});