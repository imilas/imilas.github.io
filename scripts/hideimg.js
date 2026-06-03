hexo.extend.tag.register('hideimg', function(args, content) {
  var id = 'img-' + Math.random().toString(36).substr(2, 9);
  var notHidden = false;
  args = args.filter(function(a) {
    if (a === 'not-hidden') { notHidden = true; return false; }
    return true;
  });
  var label = args.length ? args.join(' ') : 'image';
  var iconLabel = label + ' 🖼';
  var initialDisplay = notHidden ? 'block' : 'none';
  return '<a href="#" onclick="' +
    'var img=document.getElementById(\'' + id + '\');' +
    'if(!img.dataset.moved){' +
      'var p=this.closest(\'p,div,li\');' +
      'if(p&&p.parentNode){p.parentNode.insertBefore(img,p.nextSibling);}' +
      'img.dataset.moved=\'1\';' +
    '}' +
    'img.style.display=img.style.display===\'none\'||!img.style.display?\'block\':\'none\';' +
    'return false;' +
    '">' + iconLabel + '</a>' +
    '<span id="' + id + '" style="display:' + initialDisplay + ';text-align:center;">' + content + '</span>';
}, { ends: true });
