import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkCustomHeaderId from 'remark-custom-header-id';
import rehypeMermaid from "rehype-mermaid";
import tailwindcss from '@tailwindcss/vite';
import icon from "astro-icon";
import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
  site: 'https://nju-aia.github.io/',
  integrations: [ react(),icon()],

  markdown: {
    remarkPlugins: [remarkCustomHeaderId,remarkMath],
    rehypePlugins: [rehypeKatex, rehypeMermaid],
  },

  vite: {
    plugins: [tailwindcss()]
  }
});