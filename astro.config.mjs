import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkCustomHeaderId from 'remark-custom-header-id';
import rehypeMermaid from "rehype-mermaid";
import tailwindcss from '@tailwindcss/vite';
import icon from "astro-icon";
import react from '@astrojs/react';
import slidevBuilder from './tools/astro-slidev-integration.js'
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: 'https://nju-aia.github.io/',
  integrations: [ 
    react(),icon(),
    sitemap(),
    slidevBuilder({
      contentDir: 'src/content/slides',
      outDir: 'dist/slides',
      basePrefix: '/slides',
      slidevBin: 'pnpm slidev',
    }),
  ],

  markdown: {
    remarkPlugins: [remarkCustomHeaderId,remarkMath],
    rehypePlugins: [rehypeKatex, rehypeMermaid],
  },

  vite: {
    plugins: [tailwindcss()]
  }
});