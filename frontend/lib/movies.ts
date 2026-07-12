import type { Movie } from "./types"

export interface CatalogMovie extends Movie {
  id: string
  keywords: string[]
}

export const CATALOG: CatalogMovie[] = [
  {
    id: "blade-runner",
    title: "Blade Runner",
    genre: "Science Fiction, Thriller",
    decade: "1980s",
    rating: 8.1,
    poster_url: "/posters/blade-runner.png",
    overview:
      "A blade runner hunts rogue replicants through a rain-soaked neon metropolis, questioning what it means to be human.",
    keywords: ["sci-fi", "scifi", "science fiction", "noir", "dystopian", "thriller", "cyberpunk", "1980s"],
  },
  {
    id: "the-thing",
    title: "The Thing",
    genre: "Horror, Science Fiction",
    decade: "1980s",
    rating: 8.2,
    poster_url: "/posters/the-thing.png",
    overview:
      "An Antarctic research team is hunted by a shape-shifting alien that assumes the appearance of its victims.",
    keywords: ["horror", "scary", "sci-fi", "science fiction", "thriller", "paranoia", "1980s"],
  },
  {
    id: "pulp-fiction",
    title: "Pulp Fiction",
    genre: "Crime, Drama",
    decade: "1990s",
    rating: 8.9,
    poster_url: "/posters/pulp-fiction.png",
    overview:
      "The lives of two hit men, a boxer, and a gangster's wife intertwine in four tales of violence and redemption.",
    keywords: ["crime", "drama", "dark comedy", "comedy", "1990s", "gangster"],
  },
  {
    id: "eternal-sunshine",
    title: "Eternal Sunshine of the Spotless Mind",
    genre: "Romance, Science Fiction",
    decade: "2000s",
    rating: 8.3,
    poster_url: "/posters/eternal-sunshine.png",
    overview:
      "A couple undergo a procedure to erase each other from their memories, only to rediscover what they lost.",
    keywords: ["romance", "romantic", "sci-fi", "science fiction", "drama", "2000s", "melancholy"],
  },
  {
    id: "mad-max-fury-road",
    title: "Mad Max: Fury Road",
    genre: "Action, Adventure",
    decade: "2010s",
    rating: 8.1,
    poster_url: "/posters/mad-max.png",
    overview:
      "In a post-apocalyptic wasteland, a drifter and a rebel commander flee a tyrant across the desert in a roaring convoy.",
    keywords: ["action", "adventure", "post-apocalyptic", "2010s", "chase"],
  },
  {
    id: "grand-budapest",
    title: "The Grand Budapest Hotel",
    genre: "Comedy, Adventure",
    decade: "2010s",
    rating: 8.1,
    poster_url: "/posters/grand-budapest.png",
    overview:
      "A legendary concierge and his protégé become embroiled in the theft of a priceless painting and a family fortune.",
    keywords: ["comedy", "comedic", "adventure", "2010s", "whimsical", "quirky"],
  },
  {
    id: "parasite",
    title: "Parasite",
    genre: "Thriller, Drama",
    decade: "2010s",
    rating: 8.5,
    poster_url: "/posters/parasite.png",
    overview:
      "A poor family schemes to infiltrate a wealthy household, with darkly comic and shocking consequences.",
    keywords: ["thriller", "drama", "dark comedy", "comedy", "2010s", "social", "mystery"],
  },
  {
    id: "alien",
    title: "Alien",
    genre: "Horror, Science Fiction",
    decade: "1970s",
    rating: 8.5,
    poster_url: "/posters/alien.png",
    overview:
      "The crew of a commercial spacecraft is stalked and killed one by one by a deadly extraterrestrial predator.",
    keywords: ["horror", "scary", "sci-fi", "science fiction", "thriller", "1970s", "space"],
  },
  {
    id: "heat",
    title: "Heat",
    genre: "Crime, Thriller",
    decade: "1990s",
    rating: 8.3,
    poster_url: "/posters/heat.png",
    overview:
      "A relentless detective and a disciplined career thief circle each other across a sprawling, neon-lit Los Angeles.",
    keywords: ["crime", "thriller", "action", "1990s", "heist"],
  },
  {
    id: "arrival",
    title: "Arrival",
    genre: "Science Fiction, Drama",
    decade: "2010s",
    rating: 7.9,
    poster_url: "/posters/arrival.png",
    overview:
      "A linguist races to communicate with mysterious alien visitors before global tensions erupt into war.",
    keywords: ["sci-fi", "science fiction", "drama", "2010s", "cerebral", "mystery"],
  },
]
