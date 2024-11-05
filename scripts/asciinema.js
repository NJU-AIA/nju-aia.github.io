const { url } = require("hexo/dist/hexo/default_config");

hexo.extend.tag.register('asciinema', function(args) {
    const [castFile] = args; // 从参数中获取 .cast 文件名
    console.log(castFile);

    // 生成 HTML 输出
    return `
        <link rel="stylesheet" type="text/css" href="/asciinema-player/asciinema-player.css" />
        <div id="asciicast-${castFile.replace('.cast', '')}"></div>
        <script src="/asciinema-player/asciinema-player.min.js"></script>
        
        <script>
            document.addEventListener('DOMContentLoaded', () => {
                AsciinemaPlayer.create('${hexo.config.url}/${castFile}', document.getElementById('asciicast-${castFile.replace('.cast', '')}'));
            });
        </script>
    `;
}, { ends: false });
