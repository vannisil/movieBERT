import pandas as pd

data = pd.read_csv('../data/mpst_full_data.csv')
data = data.drop(labels=['imdb_id', 'split', 'synopsis_source'], axis=1)

tag_class = data["tags"].value_counts().head(15)
print("Top 15 tag classes are:", tag_class)

desired_tags = ["murder", "romantic", "violence", "psychedelic", "comedy"]
filtered_mt = data[data['tags'].isin(desired_tags)]
filtered_mt = filtered_mt.reset_index(drop=True)
filtered_mt.to_csv('../data/filtered_mt.csv', index=False)

filtered_mt['texts'] = filtered_mt['title'] + ' ' + filtered_mt['plot_synopsis']
filtered_mt = filtered_mt.drop(labels=['title', 'plot_synopsis'], axis=1)
filtered_mt = filtered_mt.rename(columns={'tags': 'labels'})
tag_mapping = {tag: str(i) for i, tag in enumerate(desired_tags)}
filtered_mt['labels'] = filtered_mt['labels'].map(tag_mapping)

print(filtered_mt)
