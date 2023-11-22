import pypandoc

# Convert Markdown to reStructuredText
# pypandoc.download_pandoc()
output = pypandoc.convert_file('input.md', 'rst', format='markdown')

with open("output.rst","w") as f:
  f.write(output)
