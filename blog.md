---
layout: page
title: Blog
permalink: /blog/
---

# Blog Posts

{% for post in site.posts %}
  <div style="margin-bottom: 2em; padding-bottom: 1em; border-bottom: 1px solid #eee;">
    <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
    <p style="color: #666; font-size: 0.9em;">{{ post.date | date: "%B %d, %Y" }}</p>
    {% if post.excerpt %}
      <p>{{ post.excerpt }}</p>
    {% endif %}
  </div>
{% endfor %}

