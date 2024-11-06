# Neighborhood-Based Collaborative Filtering   
Neighbourhood Based Collaborative Filtering are a type of recommendations algorithms based on the idea that similar users have similar behaviours in the same space of items.
   
## Collaborative Filtering Systems   
There are two type of collaborative approaches:   
- *User-Based Collaborative Filtering*: We use the ratings of neighbouring users on a set of items to compute the expected rating of a target user.   
- *Item-Based Collaborative Filtering*: We use the ratings of a user on a set of items to get the expected rating of the user on a target item.   

$$
graph LR
    D[Target Item]
    subgraph Items
      A[Item A]
      B[Item B]
      C[Item C]
    end

    User

    User-- Rate --> Items    
    Items -- Similar To --> D
    
    subgraph Calculation
        H[Weighted Average of Ratings]
    end
    
    Items --> H
    
    H -- Predicted Rating --> D

    %% Styling nodes with the provided colors
    style User fill:#F4C19B
    style A fill:#E0ACCE
    style B fill:#E0ACCE
    style C fill:#E0ACCE
    style D fill:#E0ACCE
    style H fill:#E7B3BD

$$
   
## Matrix Representation   

We can represent a collaborative filtering system using a sparse matrix. It is assumed that only a small subset of ratings is known in advance.
   

$$
(m, n) \in \mathbb{N^2} : \text {number of users} \cdot \text {number of items} \newline
R \in \mathbb{R^m \cdot R^n} : \text{ matrix of ratings } \newline

R = \begin{bmatrix}
    r_{11} & \cdots & r_{1n} \\
    \vdots  & \ddots & \vdots \\
    r_{m1}  & \cdots & r_{mn} \\
\end{bmatrix}


$$
   
Neighbourhood collaborative filtering problem can be formulated in this space by two ways:
   
- Predict the rating of a user-item combination. In this case we predict the value of the missing ratings in the matrix based on the known ratings.   
- Determine the top-k items or top-k users: In most recommendations use cases, we are not looking for the ratings values but mainly for the top k elements to recommend for each user. This formulation makes our use case easier in cardinality since there is no need to compute all ratings to get the user recommendations.   
   
## Ratings Specification   
The known ratings come from interaction of the users with the items in the real world. Depending on the use case, this ratings can be of difference nature. The most common ones are:   
1. Continuous rating: The rating is a continuous scale in a range. Such approach is relatively rare since the user have to think of a real value that corresponds to his level of like. And this being highly correlated to the understanding of the user the level of like can vary from on user to another.   
2. Interval-based ratings: The rating is a finite number of values in a range. In this space the interval between the rating is always the same. This approach also suffers from understanding of the user.   
3. Ordinal ratings: The rating is a set of ordered categorical values. Examples of such ordered categorical values can be: "Strongly Disagree", "Disagree", "Neutral", "Agree", and "Strongly Agree".   
4. Binary ratings: The rating is only two options, corresponding to positive or negative responses such as : "Like" & "Dislike".   
5. Unary ratings: The user has only one option to specify the positive preference without the ability to define a negative choice. This is the case in many of the implicit interactions that can have an end user with a website such as "Click on an item", "Add Item to cart", "View an item".   
   
## User-Based Neighborhoud Models   

$$
graph LR
    A[Target User]
    
    subgraph Users
      B[User A]
      C[User B]
      D[User C]
    end

    subgraph Items
      E[Item A]
      F[Item B]
      G[Item C]
    end
    
    Users -- Rate --> Items
    Users -- Similar User --> A

    subgraph Calculation
        H[Weighted Average of Ratings]
    end
    
    E --> H
    F --> H
    G --> H
    
    H -- Predicted Rating --> A

    %% Styling nodes with the provided colors
    style A fill:#F4C19B
    style B fill:#F4C19B
    style C fill:#F4C19B
    style D fill:#F4C19B
    style E fill:#E0ACCE
    style F fill:#E0ACCE
    style G fill:#E0ACCE
    style H fill:#E7B3BD

$$
   
In this approach, the target user rating on a specific item is defined using the ratings of its similar users. A first step is then to find the similar users i.e define a user similarity function.   
### Similarity function   
The similarity function is defined based on the observed data that we have, the ratings. One measure that captures the similarity between two users is the **[Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) **based on the covariance & the standard deviations of the ratings of the users.   

$$
\text{Let } u, v \in \mathbb{N_1^n} \text{ the user rows in the rating matrix} \newline
\text{Let } I_u , I_v \text{ the sets of ratings for users } u,v  \newline
\text{Let } I_{u \cap v} \text{ the set of items rated by both } u \text{ and } v \newline


$$

$$
\text {The mean rating  }  \bar{r}_{u \cap v} \text{ of user u in } I_{u \cap v} \text{ is : } \newline

\bar{r}_{u \cap v} = \frac
{\sum r_{uk}}
{|I_{u \cap v}|}

$$

$$
\text{And the similarity between users u and v is computed as : } \newline

\text{Sim}(u, v) =
\frac
{\sum_{i \in I_{u \cap v}} (r_{u,i} - \bar{r}_{u \cap v})(r{v,i} - \bar{r}_{v \cap u})}
{\sqrt{\sum{i \in I_{u \cap v}} (r_{u,i} - \bar{r}{u \cap v})^2}\cdot\sqrt{\sum{i \in I_{u \cap v}} (r_{v,i} - \bar{r}_{v \cap u})^2}}
$$
### Rating prediction   
The similarity function can be used to extract the closest user to the target user.   

$$
\text{Let } u,k \in \mathbb{N_1^m \cdot N_1^n} \newline
\text{Let } Sim_{u}: v \in \mathbb{N_1^m}  \mapsto Sim_{u}(v) \newline
\text {Let } P_{u,k} \text{ the subset of closest users to the user } u \text{ that rated the item } k. \newline
\text{Let } \bar{r}_u = \frac {\sum_{k \in \mathbb{N_1^n}}  r_{uk}} {|I_{u}|} \text{ the mean ratings of user } u.

$$
We can use the normalized ratings of this users on the specific item to compute the rating of the target user.
   

$$
r_{u,k} = 
\bar{r}_u + 
\frac{\sum_{v \in P_{u,k}} {Sim_{u}(v)} \cdot (r_{v,k} - \bar{r}_v)}
{\sum_{v \in P_{u,k}} Sim_u(v)}
$$
   
