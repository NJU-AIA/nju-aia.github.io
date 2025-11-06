import { readdirSync, readFileSync, mkdirSync, existsSync } from 'node:fs'
import { resolve, basename } from 'node:path'
import { execSync } from 'node:child_process'
import matter from 'gray-matter'

/**
 * @param {Object} opts
 * @param {string} opts.contentDir e.g. 'src/content/slides'
 * @param {string} opts.outDir     e.g. 'dist/slides'
 * @param {string} opts.basePrefix e.g. '/slides'
 * @param {string} opts.slidevBin  e.g. 'npx slidev' or 'pnpm slidev'
 * @param {string} [opts.config]   e.g. 'slidev.config.ts'
 */
export default function slidevBuilder (opts) {
  return {
    name: 'astro-slidev-integration',
    hooks: {
      'astro:build:done': async () => {
        const CONTENT_DIR = resolve(opts.contentDir)
        const DIST_DIR = resolve(opts.outDir)

        if (!existsSync(CONTENT_DIR)) throw new Error(`slides 内容目录不存在: ${CONTENT_DIR}`)

        mkdirSync(DIST_DIR, { recursive: true })
        const files = readdirSync(CONTENT_DIR).filter(f => f.endsWith('.md'))

        for (const file of files) {
          const abs = resolve(CONTENT_DIR, file)
          const raw = readFileSync(abs, 'utf-8')
          const { data } = matter(raw)
          const filename = basename(file, '.md')
          const slug = (data.slug || filename).trim()

          const outDir = resolve(DIST_DIR, slug)
          mkdirSync(outDir, { recursive: true })

          const args = [
            'build', abs,
            '--out', outDir,
            '--base', `${opts.basePrefix}/${slug}/`
          ]

          console.log(`🚀 构建 Slidev：${slug}`)
          execSync([opts.slidevBin, ...args].join(' '), { stdio: 'inherit' })
        }

        console.log('✅ 所有 Slidev 已构建完成')
      }
    }
  }
}
