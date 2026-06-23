import matplotlib.pyplot as plt

def sentiment_bar(sentiment_counts):
  sentiment_counts.plot(kind="bar")
  plt.title("Sentiment Distribution of BYD YouTube Comments")
  plt.xlabel("Sentiment")
  plt.ylabel("Number of Comments")
  plt.xticks(rotation=0)
  plt.show()

def sentiment_pie(sentiment_counts):
  sentiment_counts.plot(kind='pie', autopct='%1.1f%%')
  plt.title("Sentiment Percentage of BYD YouTube Comments")
  plt.ylabel("")
  plt.show()