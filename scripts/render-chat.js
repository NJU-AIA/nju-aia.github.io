hexo.extend.filter.register('after_render:html', function (htmlContent) {
  return htmlContent.replace(/\[gpt\]([\s\S]*?)\[\/gpt\]/gs, function (match, innerContent) {
    // 使用 <br> 标签分割内容
    console.log(innerContent);
    const lines = innerContent.split(/<br\s*\/?>/).map(line => line.trim()).filter(line => line);
    
    console.log(lines);
    let MessageContent = '<div class="gpt-chat">';
    lines.forEach(line => {
      // 按冒号分割说话人和消息内容
      const [speaker, message] = line.split(':');
      if (speaker && message) {
        const avatar = speaker.trim() === 'user' ? 'U' : 'AI'; // 用户头像为 "U"，AI 为 "AI"
        const speakerClass = speaker.trim().toLowerCase(); // 转换为小写，区分 user 和 bot

        MessageContent += `
          <div class="gpt-message ${speakerClass}">
            <div class="avatar">${avatar}</div>
            <div class="bubble">${message.trim()}</div>
          </div>
        `;
      }else{
        MessageContent += `
          <div class="gpt-message bot">
            <div class="bubble">${line.trim()}</div>
          </div>
         
        `;}
    });
    MessageContent += '</div>';
    
    return MessageContent;
  });
});