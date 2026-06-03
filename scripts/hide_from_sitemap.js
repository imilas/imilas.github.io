hexo.extend.filter.register('before_post_render', function(data) {
  if (data.categories && data.categories.length) {
    var hasHidden = data.categories.some(function(c) {
      return (typeof c === 'string' ? c : c.name) === 'hidden';
    });
    if (hasHidden) {
      data.sitemap = false;
      data.feed = false;
    }
  }
  return data;
});
