# How Lockdown Affected Attention Towards Video Games

## Abstract:

During the COVID-19 lockdown, people had to find other ways to keep occupied, creating a rise in video game popularity. This project aims to analyze the correlation between the severity of the lockdown with the rise and fall of various video game genres. This will be down by analyzing the language used in different Wikipedia pages for each game, as well as examining mobility data in each country to estimate the severity of lockdown.

Our project tells a fascinating story about how different types of video games faced tough challenges. We looked at the language of Wikipedia pages and connected them with data about that country's mobility change during the pandemic (for the languages that are mainly spoken in only one country). This helps us see interesting trends. Furthermore, English Wikipedia can show us what's happening worldwide as it is the most worldwide language. In a nutshell, our story shows how strict lockdowns relate to changes what video game genres people tend to play more of. It also gives us useful information about how the video game industry is changing during global disruptions and if it had a lasting effect on various genres' popularity.

### Research Questions:
- Is there a consistent trend for the same genres of video games during the COVID-19 pandemic?
  - Investigate whether certain video game genres experienced similar patterns of popularity or decline during the pandemic.
  - Explore potential factors influencing these trends, such as changes in user preferences, gaming habits, or social dynamics.

- What is the relationship between mobility patterns and the traffic of video game Wikipedia pages?
  - Examine how variations in mobility data, reflecting the severity of lockdown measures, correspond to changes in the viewership of video game Wikipedia pages.
  - Analyze whether increased or decreased mobility aligns with heightened interest in specific video game genres, indicating a potential connection between real-world restrictions and virtual entertainment.

- To what extent can English Wikipedia pages serve as a reliable estimate of the average effect on video game popularity during the pandemic?
  - Assess the generalizability of findings from English Wikipedia pages to global trends, considering cultural and linguistic differences.
  - Explore whether language-specific Wikipedia pages provide consistent insights into the popularity or decline of video game genres, or if there are notable variations that need to be considered in the analysis.

- How did the popularity change in board games compare to online games during the pandemic?
  - Is there a trend between countries that tend to have larger families with the popularity of board games in that country (Ones that have rules that can be looked up on Wikipedia).

## Additional Datasets and Methods:

To enrich our understanding, we have included additional datasets. These supplementary data sources are instrumental in examining the interesting relationship between external factors, such as the global pandemic, and the interest in different video game genres. These include all the frequencies at which Wikipedia pages for each video game in every language was accessed, especially during the COVID-19 pandemic. After having done some preprocessing and inital analysis on the dataset, we found that most of the data is useful and interesting. Although there were some missing values that we found in the dataset for certain wikipedia pages. This is all shown in the notebook. Some interesting findings that have already been done are:
- The decrease in mobility for each country that can be linked to difference in popularity of different video game genres:
![Screenshot]()

- Not only did the use of wikipedia rise, but video game related pages rose more (here is the minecraft data for example):
![Screenshot](minecraft.png)
We can see that the predicted frequency at which the minecraft wikipedia would be accessed was exceeded (the frequency is calculated as a percentage compared to the usage of wikipedia)

- The countries showed different trends which can be split into three categories:
![Screenshot](pic_1.png)

We will extend our analysis to more games and more languages, we will study different types of games (solo player, multiplayer, offline, online, campaign, battle royale, etc...) and find the ones with the highest growth of interest. To do this we will create a dataframe that will link each game to different topics (similar to the topics_linked.csv that is part of the coronawiki dataset).

All this already shows that there are very promising trends between the pandemic and video games, further analysis is needed, as detailed by these milestones:

### Internal milestones:
- Adressing the missing values that some countries have for their wikipedia traffic
- Choosing which video games are a good choice to represent a genre
- Choosing what languages are good due to them being mostly spoken in only one country
- Finding trends for each genre during the pandemic using the english wikipedia as it's an "average" since most countries speak at least some english
- Finding trends between the mobility dataset and each video game genre
- Creating the story with the findings.
- Creating the website for the datastory.
