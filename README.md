# Simon Ravé

Jekyll source for `simonrave.com`, published from the `saulane.github.io`
repository with GitHub Pages.

The site uses Minima as a GitHub Pages-compatible theme base, with local layouts
and CSS preserving the current warm, serif writing aesthetic.

## Add a Post

1. Create `_posts/YYYY-MM-DD-slug.md`.
2. Add front matter:

   ```yaml
   ---
   layout: post
   title: "Post title"
   date: YYYY-MM-DD
   description: "Short archive summary."
   permalink: /posts/slug.html
   ---
   ```

3. Write the post in Markdown. The writing index updates automatically.

For math-heavy posts, add `katex: true` to the front matter. Inline math uses
`$a^2 + b^2 = c^2$`; display math uses either `$$...$$` or a fenced `math`
code block.

## Local Preview

```sh
bundle install
bundle exec jekyll serve
```

Open `http://localhost:4000`.
