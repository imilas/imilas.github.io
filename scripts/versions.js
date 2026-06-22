hexo.extend.tag.register('versions', function(args) {
  var pairs = [];
  for (var i = 0; i < args.length; i += 2) {
    if (args[i+1]) pairs.push({ label: args[i], url: args[i+1] });
  }
  var id = 'v-' + Math.random().toString(36).substr(2, 9);

  var tabs = pairs.map(function(p, idx) {
    var activeStyle = idx === 0 ? 'font-weight:700;color:#c9cacc;' : 'font-weight:400;color:#6ba3f5;';
    return '<a href="#" class="' + id + '-tab" data-idx="' + idx + '" style="margin-right:0.6rem;text-decoration:none;' + activeStyle + '" onclick="' +
      'document.querySelectorAll(\'.' + id + '-content\').forEach(function(el){el.style.display=\'none\'});' +
      'document.getElementById(\'' + id + '-\'+this.dataset.idx).style.display=\'block\';' +
      'document.querySelectorAll(\'.' + id + '-tab\').forEach(function(el){el.style.fontWeight=\'400\';el.style.color=\'#6ba3f5\'});' +
      'this.style.fontWeight=\'700\';this.style.color=\'#c9cacc\';' +
      'return false;' +
      '">[' + p.label + ']</a>';
  }).join('');

  var contents = pairs.map(function(p, idx) {
    var isPdf = /\.pdf$/i.test(p.url);
    var media = isPdf
      ? '<embed src="' + p.url + '" type="application/pdf" style="max-width:500px;width:100%;height:600px;">'
      : '<img src="' + p.url + '" style="max-width:500px;width:100%;height:auto;">';
    var initialDisplay = idx === 0 ? 'block' : 'none';
    return '<div id="' + id + '-' + idx + '" class="' + id + '-content" style="display:' + initialDisplay + ';">' + media + '</div>';
  }).join('');

  return '<div style="max-width:500px;width:100%;margin:0 auto;text-align:center;"><div style="text-align:left;">' + tabs + '</div>' + contents + '</div>';
});
