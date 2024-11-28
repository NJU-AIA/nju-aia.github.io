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
        MessageContent += `
          <div class="gpt-message ${speaker.trim()}">
            <p>${message.trim()}</p>
          </div>
        `;
      }else{
        MessageContent += `
          <div class="gpt-message">
            <p>${line.trim()}</p>
          </div>
        `;}
    });
    MessageContent += '</div>';
    
    return MessageContent;
  });
});