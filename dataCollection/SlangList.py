
class SlangList:

    def __init__(self, drugType):
        self.drugType = drugType

        # get list of opium names and the string of street names
        self.opiumList = ["opium", "opiate", "opiates", "opioid"]
        self.opiumStreet = "Auntie; Aunt Emma; Big O; Black; Black Russian; Chandoo; China;\
        Chinese Molasses; Chinese Tobacco; Chocolate; Cruz; Dopium; Dover’s Powder; \
        Dream Gum; Dream Stick; Dreams; Easing Powder; God’s Medicine; Goma; Gondola; \
        Goric; Great Tobacco; Gum; Hocus; Hops; Incense; Joy Plant; Midnight Oil; Opio;  \
        Pen Yan; Pin Gon; Pin Yen; Pox; Skee; Toxy; Toys; When-Shee; Zer"

        self.cocaineList = ["cocaine"]
        self.cocaineStreet = "777; A-1; All-American Drug; Angel Powder; Angie; Animals; Audi;" \
                             " Aunt Nora; Azucar; Baby Powder; Barrato; Basuco; Bazooka; Beach; " \
                             "Belushi (mixed with heroin); Bernie’s Flakes; Bernie’s Gold Dust; Big Bloke; Big C; Big Flake; " \
                             "Big Rush; Billie Hoke; Bird; Birdie Powder; Blanca Nieves; Blanco; Blast; Blizzard; " \
                             "Blonde; Blocks; Blow; BMW; Bobo; Bolitas; Bolivian Marching Powder; Bombita Bouncing Powder;" \
                             " Brisa; C-Dust; Caca; Cadillac; California Pancakes; Calves; Candy; Car; " \
                             "Carney; Carrie Nation; Cars; Case; Cebolla; Cecil; Cement; Charlie; Chevy; Cheyenne; " \
                             "Chinos; Chiva; Clear Kind; Clear Tires; Coca; Coca-Cola; Cocazo; Coconut; Coke; Cola; " \
                             "Colorado; Connie; Cookie; Crow; Crusty Treats; Cuadro; Death Valley; Designer Jeans; D" \
                             "evil’s Dandruff; Diente; Dienton; Dona Blanca; Double Bubble; Dove; Dream; Dulces; " \
                             "Duracell; Dust; Escama; Escorpino; Falopa; Fish (liquid cocaine); Flake; Flea Market Jeans; " \
                             "Florida Snow; Flour; Food; Foolish Powder; Fox; Freeze; Friskie Powder; Frula; Gabacho; " \
                             "Galaxy; Gallos; Gato; Gift of the Sun; Gin; Girl; Girlfriend; Glad Stuff; Gold Dust; " \
                             "Green Gold; Gringa; Grout; Gueros; Guitar; Hamburger; Happy Dust; Happy Powder; Happy Trails;" \
                             " Heaven; Heaven Dust; Henry VIII; Hooter; Hundai; Hunter; Icing; Inca Message; Izzy; Jam; " \
                             "Jeep; Jelly; John Deere; Joy Flakes; Joy Powder; Junk; King’s Habit; Kordell; Lady; Lady Snow; " \
                             "Late Night; Lavada; Leaf; Line; Loaf; Love Affair; Maca Flour; Mama Coca; Mandango; Maradona; " \
                             "Mayo; Melcocha; Mercedes; Milk; Milonga; Mojo; Mona Lisa; Mosquitos; Movie Star Drug; Mujer; " \
                             "Napkin; Nieve; Niña; Nose Candy; Nose Powder; Old Lady; Oyster Stew; Paint; Paloma; Palomos; " \
                             "Pantalones; Papas; Paradise; Paradise White; Parrot; Pearl; Pedrito; Perico; Peruvian; " \
                             "Peruvian Flake; Peruvian Lady; Pescado; Pez; Pillow; Pimp; Pollo; Polvo; Powder; Powder Diamonds;" \
                             " Puritain; Queso Blanco; Racehorse Charlie; Rambo; Refresco; Refrescas; Reindeer Dust; Rims; " \
                             "Rocky Mountain; Rolex; Rooster; Scale; Schmeck; Schoolboy; Scorpion; Scottie; Seed; Serpico; " \
                             "Sierra; Shirt; Ski Equipment; Sleigh Ride; Snow; Snow Bird; Snow Cone; Snow White; Snowball; " \
                             "Snowflake; Society High; Soda; Soditas; Soft; Space (mixed with PCP); Speedball; Stardust; " \
                             "Star Spangled Powder; Studio Fuel; Suave; Sugar; Superman; Sweet Stuff; Talco; Talquito; Tamales; " \
                             "Taxi; Tecate; Teenager; Teeth; Tequila; Thunder; Tire; Tonto; Toot; Tortes; Toyota; T-Shirts; " \
                             "Turkey; Tutti-Frutti; Vaquita; Wash; Wet; Whack (mixed with PCP); White; White Bitch; " \
                             "White Cross; White Girl; White Goat; White Horse; White Lady; White Mercedes Benz; White Mosquito;" \
                             " White Paint; White Powder; White Root; White Shirt; White T; Whitey; Whiz Bang; Wings; Wooly; " \
                             "Work; Yayo; Yeyo; Yoda; Zip"

        self.heroinList = ["heroin"]
        self.heroinStreet = "A-Bomb; Achivia; Adormidera; Antifreeze; Aunt Hazel; Avocado; Azucar; Bad Seed;Ballot; Basketball; " \
                            "Basura; Beast; Beyonce; Big Bag; Big H; Big Harry; Bird; Birdie Powder; Black; BlackBitch; " \
                            "Black Goat; Black Olives; Black Paint; Black Pearl; Black Sheep; Black Tar; Blanco; Blue; Blow Dope;" \
                            "Blue Hero; Bombita (mixed with cocaine); Bombs Away; Bonita; Boy; Bozo; Brea Negra; Brick Gum; Brown;" \
                            "Brown Crystal; Brown Rhine; Brown Sugar; Bubble Gum; Burrito; Caballo; Caballo Negro; Caca; Café; " \
                            "CapitalH; Carga; Caro; Cement; Chapopote; Charlie; Charlie Horse; Cheese; Chicle; Chiclosa; China; " \
                            "China Cat;China White; Chinese Food; Chinese Red; Chip; Chiva; Chiva Blanca; Chivones; Chocolate; " \
                            "Chocolate Balls;Choko; Chorizo; Chutazo; Coco; Coffee; Comida; Crown Crap; Curley Hair; Dark; Dark Girl; " \
                            "Dead on Arrival(DOA); Diesel; Diesel; Dirt; Dog Food; Doggie; Doojee; Dope; Dorado; Down; Downtown; " \
                            "Dreck; Dynamite;Dyno; El Diablo; Engines; Fairy Dust; Flea Powder; Foolish Powder; Galloping Horse; " \
                            "Gamot; Gato; George Smack; Girl; Golden Girl; Good & Plenty; Good H; Goma; Gorda; Gras; Grasin; Gravy; " \
                            "Gum; H; H-Caps; Hairy; Hard Candy; Harry; Hats; Hazel; Heaven Dust; Heavy; Helen; Helicopter; " \
                            "Hell Dust; Henry; Hercules; Hero; Him; Hombre; Horse; Hot Dope; Hummers; Jojee; Joy Flakes; " \
                            "Joy Powder; Junk; Kabayo; Karachi; Karate; King’s Tickets; Lemonade; Lenta; Lifesaver; Manteca; " \
                            "Marias; Mayo; Mazpan; Meal; Menthol; Mexican Brown; Mexican Horse; Mexican Mud; Mexican Treat; " \
                            "Modelo Negra; Mojo; Mole; Mongega; Morena; Morenita; Mortal Combat; Motors; Mud; Mujer; Muzzle; " \
                            "Nanoo; Negra; Negra Tomasa; Negrita; Nice and Easy; Night; Noise; Obama; Old Steve; Pants; Patty; " \
                            "Peg; P-Funk; Piezas; Plata; Poison; Polvo; Poppy; Powder; Prostituta Negra; Puppy; Pure; Rambo; " \
                            "Red Chicken; Red Eagle; Reindeer Dust; Roofing Tar; Sack; Salt; Sand; Scag; Scat; Schmeck; Sheep; " \
                            "Shirts; Shoes; Skag; Slime; Smack; Smeck; Snickers; Speedball (mixed with cocaine); Spider Blue; " \
                            "Sticky Kind; Stufa; Sugar; Sweet Jesus; Tan; Tar; Tecata; Tires; Tootsie Roll; Tragic Magic; Trees; " \
                            "Turtle; Vidrio; Whiskey; White; White Boy; White Girl; White Junk; White Lady; White Nurse; White Shirt;" \
                            " White Stuff; Wings; Witch; Witch Hazel; Zapapote"

        self.marijuanaList = ["marijuana"]
        self.marijuanaStreet = "420; Acapulco Gold; Acapulco Red; Ace; African Black; African Bush; Airplane; Alfombra; " \
                               "Alice B Toklas; All-Star; Angola; Animal Cookies; Arizona; Ashes; Aunt Mary; " \
                               "Baby; Bale; Bambalachacha; Barbara Jean; Bareta; Bash; BC Budd; Bernie; Bhang; Big Pillows; " \
                               "Biggy; Black Bart; Black Gold; Black Maria; Blondie; Blue Cheese; Blue Crush; Blue Jeans; Blue Sage; " \
                               "Blueberry; Bobo Bush; Boo; Boom; Broccoli; Bud; Budda; Burritos Verdes; Bush; Cabbage; Cali; " \
                               "Canadian Black; Catnip; Cheeba; Chernobyl; Cheese; Chicago Black; Chicago Green; Chippie; Chistosa; " \
                               "Christmas Tree; Chronic; Churo; Cigars; Citrol; Cola; Colorado Cocktail; Cookie; " \
                               "Cotorritos; Crazy Weed; Creeper Bud; Crippy; Crying Weed; Culican; Dank; Dew; Diesel; Dimba; " \
                               "Dinkie Dow; Dirt Grass; Ditch Weed; Dizz; Djamba; Dody; Dojo; Domestic; Donna Juana; Doobie; " \
                               "Downtown Brown; Drag Weed; Dro (hydroponic); Droski (hydroponic); Dry High; Endo; Fine Stuff; " \
                               "Fire; Flower; Flower Tops; Fluffy; Fuzzy Lady; Gallito; Garden; Gauge; Gangster; Ganja; Gash; " \
                               "Gato; Ghana; Gigi (hydroponic) Giggle Smoke; Giggle Weed; Girl Scout Cookies (hydroponic); " \
                               "Gloria; Gold; Gold Leaf; Gold Star; Gong; Good Giggles; Gorilla; Gorilla Glue; Grand Daddy Purp; " \
                               "Grass; Grasshopper; Green; Green-Eyed Girl; Green Eyes; Green Goblin; Green Goddess; " \
                               "Green Mercedes Benz; Green Paint; Green Skunk; Grenuda; Greta; Guardada; Gummy Bears; Gunga; " \
                               "Hairy Ones; Hash; Hawaiian; Hay; Hemp; Herb; Hierba; Holy Grail; Homegrown; Hooch; Humo; Hydro; " \
                               "Indian Boy; Indian Hay; Jamaican Gold; Jamaican Red; Jane; Jive; Jolly Green; Jon-Jem; Joy Smoke; " \
                               "Juan Valdez; Juanita; Jungle Juice; Kaff; Kali; Kaya; KB; Kentucky Blue; KGB; Khalifa; Kiff; Killa; " \
                               "Kilter; King Louie; Kona Gold; Kumba; Kush; Laughing Grass; Laughing Weed; Leaf; Lechuga; Lemon-Lime; " \
                               "Liamba; Lime Pillows; Little Green Friends; Little Smoke; Loaf; Lobo; Loco Weed; Love Nuggets; " \
                               "Love Weed; M.J.; Machinery; Macoña; Mafafa; Magic Smoke; Manhattan Silver; Maracachafa; Maria; " \
                               "Marimba; Mariquita; Mary Ann; Mary Jane; Mary Jones; Mary Warner; Mary Weaver; Matchbox; Matraca; " \
                               "Maui Wowie; Meg; Method; Mexican Brown; Mexican Green; Mexican Red; Mochie (hydroponic); Moña; Monte; " \
                               "Moocah; Mootie; Mora; Morisqueta; Mostaza; Mota; Mother; Mowing the Lawn; Muggie; Narizona; " \
                               "Northern Lights; O-Boy; O.J.; Owl; Paja; Panama Cut; Panama Gold; Panama Red; Pakalolo; Palm; Paloma; " \
                               "Parsley; Pelosa; Phoenix; Pillow; Pine; Platinum Cookies (hydroponic); Platinum Jack; Pocket Rocket; " \
                               "Popcorn; Pot; Pretendo; Puff; Purple Haze; Queen Ann’s Lace; Ragweed; Railroad Weed; Rainy Day Woman; " \
                               "Rasta Weed; Red Cross; Red Dirt; Reefer; Reggie; Repollo; Righteous Bush; Root; Rope; Rosa Maria; " \
                               "Salt & Pepper; Santa Marta; Sasafras; Sativa; Sinsemilla; Shmagma; Shora; Shrimp; Shwag; Skunk; " \
                               "Skywalker (hydroponic); Smoke; Smoochy Woochy Poochy; Smoke; Smoke Canada; Spliff; Stems; Stink Weed; " \
                               "Sugar Weed; Sweet Lucy; Tahoe (hydroponic); Tex-Mex; Texas Tea; Tila; Tims; Tosca; Trees; Tweeds; " \
                               "Wacky Tobacky; Wake and Bake; Weed; Weed Tea; Wet (mixed with PCP); Wheat; White-Haired Lady; " \
                               "Wooz; Yellow Submarine; Yen Pop; Yerba; Yesca; Young Girls; Zacate; Zacatecas; Zambi; Zoom"




    def removeBlank(self, origList):
        for i in range(len(origList)):
            origList[i] = origList[i].strip()
        return origList

    def convertToList(self, str):
        strLower = str.lower()
        strList = strLower.strip().split(";")
        strList = self.removeBlank(strList)
        return strList

    def getDrugList(self, drugType):
        switcher = {
            "opium": self.opiumList,
            "cocaine": self.cocaineList,
            "heroin": self.heroinList,
            "marijuana": self.marijuanaList
        }
        return switcher.get(drugType, "error input!")


    #def getOpiumStreetList(self):
    #    return self.convertToList(self.opiumStreet)

    def getDrugStreetList(self, drugType):
        switcher = {
            "opium": self.convertToList(self.opiumStreet),
            "cocaine": self.convertToList(self.cocaineStreet),
            "heroin": self.convertToList(self.heroinStreet),
            "marijuana": self.convertToList(self.marijuanaStreet)
        }
        return switcher.get(drugType, "error input!")
