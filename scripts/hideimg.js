hexo.extend.tag.register('hideimg', function(args, content) {
  var id = 'img-' + Math.random().toString(36).substr(2, 9);
  var linkId = 'link-' + id;
  var label = args.length ? args.join(' ') : 'image';
  var showText = 'show ' + label;
  var hideText = 'hide ' + label;
  var showTextJs = showText.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
  var hideTextJs = hideText.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
  return '<span class="hide-img-wrapper">' +
    '<a href="#" id="' + linkId + '" class="show-img-link" onclick="' +
      'var img=document.getElementById(\'' + id + '\');' +
      'var lnk=document.getElementById(\'' + linkId + '\');' +
      'if(img.style.display===\'none\'){img.style.display=\'block\';lnk.textContent=\'' + hideTextJs + '\';}' +
      'else{img.style.display=\'none\';lnk.textContent=\'' + showTextJs + '\';}' +
      'return false;' +
    '">' + showText + '</a>' +
    '<span id="' + id + '" style="display:none;">' + content + '</span>' +
    '</span>';
}, { ends: true });
