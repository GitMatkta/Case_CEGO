"""
Step 1: Generate synthetic Danish chat room messages programmatically.
Uses curated templates with randomization for realistic variety.
"""
import json, random, csv
from datetime import datetime, timedelta
from collections import Counter

random.seed(42)

CLEAN = [
    "Hej alle sammen! 😊","Godaften derude!","Hej! Er der nogen online?",
    "Hyggeligt at se jer herinde igen","God weekend til jer alle!",
    "Hej med jer, hvad sker der?","Åh hej, længe siden!",
    "Nogen der har det godt i aften?","Hejsa! Klar til en hyggelig aften",
    "Godmorgen! Tidlig fugl i dag 😄",
    "Har I prøvet den nye slot? Den ser fed ud","Nogen der kan anbefale et godt spil?",
    "Blackjack er min favorit, hvad med jer?","Roulette er sjovt men jeg holder mig til lavt indskud",
    "Den nye bonus er ret god faktisk","Hvad er reglerne for det her turneringsspil?",
    "Nogen der har prøvet live dealer?","Jeg foretrækker bordspil frem for slots",
    "Spillede lidt i går, det var hyggeligt","Hvad er jeres yndlingsspil herinde?",
    "Kender I et godt begynderspil?","Jackpotten ser vild ud i dag!",
    "Er der nogen der har prøvet det nye theme?","Jeg synes grafikkerne er blevet meget bedre",
    "Fedt, vandt lige et frispin! 🎰","Nice, 200 kr op — god start på aftenen",
    "Fik en ok gevinst i dag, happy 😊","Lille gevinst men det er altid sjovt",
    "Grattis til dig! Godt gået!","Tillykke med gevinsten! 🎉",
    "Haha det var heldigt!","Wauw, det var et godt spin!",
    "Så I kampen i går? Sindssygt mål!","Vejret er helt elendigt i dag, perfekt til at hygge inde",
    "Fodbold i weekenden, nogen der skal se?","Danmark spiller i morgen!",
    "Det regner så meget, håber det stopper snart","Endelig sol!",
    "Lige lavet aftensmad, nu er det hyggetid","Nogen der har en god opskrift på pasta?",
    "Tager en kop kaffe, brb","Laver lige popcorn, et øjeblik 🍿",
    "Skal hente børnene snart, ses senere!","Kører lige en tur i Netto, er tilbage om lidt",
    "Hvordan virker den der free spin bonus?","Er chatten nede for andre?",
    "Kan man spille fra mobilen?","Hvornår lukker chatten?",
    "Får I også den der fejlmeddelelse?","Min forbindelse er lidt langsom i aften",
    "Haha 😂","Det er så rigtigt!","Enig!","LOL","Nå okay 😅",
    "Det lyder godt!","Spændende!","Ja det tror jeg også",
    "Nej det har jeg ikke prøvet","Måske, vi får se!",
    "Det er en god pointe","Interessant!","Helt enig med dig der",
    "Cool cool","Nice!","👍","Det er fair nok",
    "Jeg er lidt ny herinde, hej!","Kan nogen hjælpe mig?",
    "Tak for tippet!","Det prøver jeg lige","Super, tak!",
    "Jeg sætter altid en grænse inden jeg starter","I dag holder jeg mig til 200 kr max",
    "Husk at spille ansvarligt folkens 😊","Jeg spiller kun for sjov, ikke for pengene",
    "Tid til at stoppe for i dag, det har været sjovt!","Min grænse er nået, ses i morgen!",
    "Man skal vide hvornår man skal stoppe","Jeg har sat en timer på 1 time",
]

OFF_LOW = [
    "Lort, tabte igen 😅","For fanden da, det spin var dårligt",
    "Satans, det var tæt på!","Argh, det er bare typisk",
    "Hold da op, det var et dårligt spin","Pis, det gik helt galt",
    "Fandens også, næsten!","Øv for søren da",
    "Det der spil er sku da underligt","Sgu da irriterende",
    "Ej for helvede, igen?!","Det passer mig sku ikke",
    "Av av av, det var dyrt 😬","Pokkers, troede jeg havde den",
    "Kraft edeme det var tæt","Det var sku nansen da",
    "Årh lort mansen","For satan da","Pis og papir",
    "Ej det er kraftedeme irriterende","Sgu da noget skidt",
    "Damn det var uheldigt","Shit mansen","Nå for pokker",
    "Det er sku noget møg det her","Åh nej, ikke igen...",
]

OFF_MED = [
    "Hold kæft du er dårlig til det her","Er du dum eller hvad?",
    "Du aner sku da ikke hvad du snakker om","Hvor er du irriterende, stop",
    "Folk herinde er jo helt idioter","Hvad fanden bilder du dig ind?",
    "Kan du ikke bare holde din kæft?","Du er så pinlig at høre på",
    "Hvad er det for noget pjat du lukker ud?","Er du helt væk eller hvad?",
    "Hold nu op med dit pis","Du er jo totalt håbløs",
    "Kan I ikke bare stoppe, I er så dumme","Ej seriøst, hold kæft med det der",
    "Taber. Lær at spille ordentligt","Du ved tydeligvis ingenting",
    "Sikke en omgang vrøvl du lukker ud","Du burde slet ikke spille",
    "Typisk amatør, gå hjem","Du er den værste spiller herinde",
    "Stop dit bavl","Idiot. Bare stop.","Hvor er det lamt det du siger",
    "Gå dog væk hvis du ikke kan finde ud af det","Du ødelægger stemningen herinde",
]

OFF_HIGH = [
    "Jeg finder dig og smadrer dig","Du fortjener en lussing din idiot",
    "Pas på dig selv, jeg ved hvem du er","Jeg giver dig tæsk hvis vi mødes",
    "Din dumme so, hold kæft","Dit stykke lort, forsvind herfra",
    "Jeg håber du taber alt du ejer","Du er det mest patetiske menneske herinde",
    "Hvis jeg møder dig, er du færdig","Skrid herfra din tabertypes",
    "Du burde skamme dig, dit svin","Hold din kæft eller der sker noget",
    "Jeg smadrer alt hvad du ejer","Du er intet værd, forsvind",
    "Prøv at sige det til mit ansigt din kujon","Jeg ved hvor du bor",
    "Din pathtetiske taber, skrid","Jeg slår dig ned næste gang",
    "Du er det mest ulækre menneske jeg kender","Skrid af helvede til",
]

GAM_LOW = [
    "Jeg prøver lige at vinde det tilbage","Bare ét spin mere, hehe",
    "Har spillet lidt mere end planlagt i dag 😅","Prøver lige lidt mere, det vender snart",
    "Jeg tror jeg har heldet med mig i aften, kører lidt videre",
    "Tabte lidt i går men prøver igen i dag","Bare en lille smule mere...",
    "Det plejer at vende, jeg spiller videre","Nå men ét spin mere kan vel ikke skade",
    "Jeg blev lidt længere end planlagt haha","Hmm har brugt lidt mere end budgetteret",
    "Tabte 500 kr men det er fint, spiller lidt mere","Jeg vinder det nok igen i morgen",
    "Har haft en dårlig streak men det vender","Spiller lige lidt mere, føler den kommer",
    "Tror mit held vender snart","Bare 100 kr mere, så stopper jeg",
    "Skulle egentlig stoppe men...","Prøver lige et par stykker mere",
    "Det her er min aften, kan jeg mærke det!","Taber lidt men det er en del af spillet",
    "Satser lidt mere i dag, har en god følelse","Ej prøver lige én gang til",
    "Vandt i går, tabte i dag, sådan er det jo","Lidt mere og så stopper jeg",
]

GAM_MED = [
    "Har tabt 5.000 denne uge, prøver at vinde det tilbage",
    "Lånte penge af min ven til at spille videre","Kan sgu ikke stoppe, det SKAL vende",
    "Har brugt hele min løn allerede og det er kun d. 10",
    "Min konto er i minus men jeg kan mærke det kommer",
    "Har spillet i 8 timer i streg nu","Jeg burde stoppe men kan ikke",
    "Tabte 10.000 denne måned, dobler op nu",
    "Solgte min gamle telefon for at spille mere",
    "Har ikke sovet fordi jeg har spillet hele natten",
    "Tager et lån for at dække mine tab","Brugte sparepengene, men det er fint",
    "Føler mig nødt til at fortsætte, har tabt for meget",
    "Min bankkonto ser ikke godt ud men næste spin bliver stort",
    "Bruger mere tid på at spille end på arbejde ærligt talt",
    "Har spillet op det jeg skulle bruge på regninger",
    "Låner af kassekredit for at spille videre",
    "Har øget min indsats markant for at indhente tabene",
    "Tredje dag i streg jeg sidder her, men det vender",
    "Kan ikke tænke på andet end at spille lige nu",
]

GAM_HIGH = [
    "Har tabt huslejen, ved ikke hvad jeg skal gøre",
    "Min kone aner ikke hvor mange penge jeg har tabt",
    "Jeg skjuler det for alle, ingen må vide det",
    "Har tabt over 50.000 kr de sidste to måneder, er desperat",
    "Jeg kan ikke betale mine regninger mere pga spil",
    "Har solgt ting fra huset uden min kæreste ved det",
    "Jeg har brug for at vinde, det er min eneste udvej",
    "Har taget forbrugslån for at spille, det er ude af kontrol",
    "Tænker på at spille 24/7, kan ikke fokusere på noget",
    "Har tabt alt hvad jeg har sparet op over 5 år",
    "Mine børn får ikke gaver til jul fordi jeg har spillet pengene",
    "Jeg lyver for min familie om hvor pengene er gået",
    "Har ikke betalt husleje i 2 måneder, spillet det hele op",
    "Føler mig fuldstændig ude af kontrol, men spiller videre",
    "Har overvejet at låne af kriminelle for at dække mine tab",
    "Min kæreste har truet med at forlade mig pga mit spil",
    "Jeg er så desperat at jeg spiller med penge jeg ikke har",
    "Alt er fucked, men måske vinder jeg det hele tilbage i aften",
    "Kan ikke sove, kan ikke spise, tænker kun på at spille",
    "Har mistet venner og familie over det her, men stopper ikke",
]

BOTH_L = [
    "Lort mansen, tabte 500 kr igen men prøver lige videre",
    "For fanden, det vender ALDRIG, satser lige lidt mere",
    "Sgu irriterende, men spiller videre til jeg har vundet det tilbage",
    "Pis, tabte igen, bare ét mere spin for satan",
    "Det er sku da noget møg, men giver ikke op",
    "Argh for helvede, prøver lige lidt mere",
    "Satans, tabte lidt, men det kommer igen, spiller videre",
]

BOTH_M = [
    "Hold kæft I er irriterende herinde, og jeg har tabt 8000 kr i aften",
    "Er du dum? Jeg har tabt det hele og du sidder og griner",
    "I kan rende mig, har brugt hele min løn herinde",
    "Dit pis af et spil, og nu kan jeg ikke stoppe",
    "Hold kæft hvor er det her dumt, men jeg SKAL vinde det tilbage",
    "Idioter alle sammen, og jeg er den største idiot der stadig spiller",
    "Skrid med jeres gode råd, jeg har tabt 10.000 og det er min business",
]

BOTH_H = [
    "Jeg finder dig og smadrer dig! Har tabt huslejen pga det her lorte spil",
    "Hold din kæft din taber! Jeg har tabt alt og er desperat",
    "Jeg slår nogen ned hvis jeg ikke vinder snart, har intet tilbage",
    "Dit stykke lort, det er jeres skyld jeg har tabt alt",
    "Jeg har mistet alt og I sidder og griner, I skal passe på",
    "Skrid af helvede til alle sammen, jeg har tabt 50.000 og kan ikke stoppe",
    "Jeg hader jer alle, har solgt alt for at spille og nu er alt væk",
]

EDGE_CASES = [
    {"content":"Hold kæft det var et godt spin! 🎰","cat":"clean","sev":"none"},
    {"content":"Ej det er jo sygt godt det her!","cat":"clean","sev":"none"},
    {"content":"Haha jeg er verdens værste spiller 😂","cat":"clean","sev":"none"},
    {"content":"Det her spil DRÆBER mig 😅 (i den gode forstand)","cat":"clean","sev":"none"},
    {"content":"Ej shut up det er for vildt! 🤩","cat":"clean","sev":"none"},
    {"content":"Hold kæft mansen, sikke en gevinst!","cat":"clean","sev":"none"},
    {"content":"Det er jo helt crazy det her, elsker det!","cat":"clean","sev":"none"},
    {"content":"Nå men så slår jeg mig selv ihjel... med det her spil hele aftenen 😄","cat":"clean","sev":"none"},
    {"content":"Jeg satser stort på at Danmark vinder i morgen!","cat":"clean","sev":"none"},
    {"content":"All in på at vejret bliver godt i weekenden ☀️","cat":"clean","sev":"none"},
    {"content":"Vandt 500 kr! Lægger dem lige til side","cat":"clean","sev":"none"},
    {"content":"Tabte 100 kr men det var sjovt, nu stopper jeg","cat":"clean","sev":"none"},
    {"content":"Sætter 200 kr af til i aften og ikke en krone mere","cat":"clean","sev":"none"},
    {"content":"Jeg spiller kun med det jeg har råd til at tabe","cat":"clean","sev":"none"},
    {"content":"Nok for i dag! God disciplin er vigtigt","cat":"clean","sev":"none"},
    {"content":"Haha nogen sagde 'hold kæft' som kompliment, klassisk dansk","cat":"clean","sev":"none"},
    {"content":"Min ven siger altid 'for satan det er godt' når han vinder","cat":"clean","sev":"none"},
    {"content":"Det her spil er virkelig frustrerende altså","cat":"clean","sev":"none"},
    {"content":"Jeg er ret irriteret lige nu tbh","cat":"clean","sev":"none"},
    {"content":"Det er lidt ærgerligt at tabe men okay","cat":"clean","sev":"none"},
    {"content":"Har spillet meget i weekenden, sjovt nok!","cat":"clean","sev":"none"},
    {"content":"Brugte 1000 kr i dag, det var min budget, fair nok","cat":"clean","sev":"none"},
    {"content":"Prøver lige én gang mere og så er det slut","cat":"gambling","sev":"low"},
    {"content":"Spiller lidt mere end normalt men har råd","cat":"clean","sev":"none"},
    {"content":"For helvede det her er noget lort!","cat":"offensive","sev":"low"},
    {"content":"Satans til dårligt spil altså","cat":"offensive","sev":"low"},
    {"content":"Er der nogen der kender til hjælp mod ludomani?","cat":"clean","sev":"none"},
    {"content":"Husk man kan altid ringe til StopSpillet","cat":"clean","sev":"none"},
    {"content":"Pas på jer selv derude, spil ansvarligt ❤️","cat":"clean","sev":"none"},
    {"content":"Har I set den nye ansvarligt spil-funktion? Fed feature","cat":"clean","sev":"none"},
]

def vary(t):
    fns = [
        lambda t:t, lambda t:t,
        lambda t:t+" "+random.choice(["","haha","lol","tbh","altså","jo","nå"]),
        lambda t:t.lower() if random.random()>0.5 else t,
        lambda t:t.rstrip("!?.") + random.choice(["!","!!","...","."]),
        lambda t:t+" "+random.choice(["😊","😅","😂","🤔","😤","💪","🎰",""]),
    ]
    return random.choice(fns)(t).strip()

def main():
    msgs = []
    mid = 1
    bt = datetime(2026,4,13,10,0,0)

    cfgs = [
        (1,CLEAN,"clean","none",55),(2,CLEAN,"clean","none",50),
        (3,CLEAN,"clean","none",50),(4,CLEAN,"clean","none",50),
        (5,CLEAN,"clean","none",50),(6,CLEAN,"clean","none",50),
        (7,CLEAN,"clean","none",50),(8,CLEAN,"clean","none",50),
        (9,CLEAN,"clean","none",45),(10,CLEAN,"clean","none",45),
        (11,CLEAN,"clean","none",45),(12,CLEAN,"clean","none",45),
        (13,OFF_LOW,"offensive","low",45),(14,OFF_LOW,"offensive","low",45),
        (15,OFF_LOW,"offensive","low",35),(16,OFF_LOW,"offensive","low",35),
        (17,OFF_MED,"offensive","medium",40),(18,OFF_MED,"offensive","medium",40),
        (19,OFF_HIGH,"offensive","high",30),(20,OFF_HIGH,"offensive","high",30),
        (21,GAM_LOW,"gambling","low",40),(22,GAM_LOW,"gambling","low",40),
        (23,GAM_LOW,"gambling","low",40),
        (24,GAM_MED,"gambling","medium",35),(25,GAM_MED,"gambling","medium",35),
        (26,GAM_HIGH,"gambling","high",30),(27,GAM_HIGH,"gambling","high",30),
        (28,BOTH_L,"both","low",25),(29,BOTH_M,"both","medium",25),
        (30,BOTH_H,"both","high",20),
    ]

    for uid,pool,cat,sev,n in cfgs:
        do = random.randint(0,6)
        hs = random.randint(10,21)
        ub = bt + timedelta(days=do, hours=hs-10)
        mix = cat != "clean"
        cr = 0.3 if sev=="low" else (0.2 if sev=="medium" else 0.15)
        for i in range(n):
            if mix and random.random()<cr:
                c,ca,sv = random.choice(CLEAN),"clean","none"
            else:
                c,ca,sv = random.choice(pool),cat,sev
            c = vary(c)
            ts = ub + timedelta(minutes=i*random.randint(2,12), seconds=random.randint(0,59))
            msgs.append({"msg_id":mid,"user_id":uid,"content":c,
                         "timestamp":ts.strftime("%Y-%m-%d %H:%M:%S"),
                         "category":ca,"severity":sv})
            mid += 1

    for i,ec in enumerate(EDGE_CASES):
        uid = 31+(i%5)
        ts = bt + timedelta(days=random.randint(0,6),hours=random.randint(0,10),
                            minutes=random.randint(0,59),seconds=random.randint(0,59))
        msgs.append({"msg_id":mid,"user_id":uid,"content":vary(ec["content"]),
                     "timestamp":ts.strftime("%Y-%m-%d %H:%M:%S"),
                     "category":ec["cat"],"severity":ec["sev"]})
        mid += 1

    msgs.sort(key=lambda x:x["timestamp"])
    for i,m in enumerate(msgs,1): m["msg_id"]=i

    op = "E:\Code\homeProjects\CEGO\FTIDF_LogisticRegression\chat_messages.json"
    with open(op,"w",encoding="utf-8") as f: json.dump(msgs,f,ensure_ascii=False,indent=2)

    cp = "E:\Code\homeProjects\CEGO\FTIDF_LogisticRegression\chat_messages.csv"
    with open(cp,"w",encoding="utf-8",newline="") as f:
        w = csv.DictWriter(f,fieldnames=["msg_id","user_id","content","timestamp","category","severity"])
        w.writeheader(); w.writerows(msgs)

    cats = Counter(m["category"] for m in msgs)
    sevs = Counter(m["severity"] for m in msgs)
    print(f"Total messages: {len(msgs)}")
    print(f"Unique users:   {len(set(m['user_id'] for m in msgs))}")
    print(f"\nCategory distribution:")
    for k,v in sorted(cats.items()): print(f"  {k:12s}: {v:4d}  ({v/len(msgs)*100:.1f}%)")
    print(f"\nSeverity distribution:")
    for k,v in sorted(sevs.items()): print(f"  {k:8s}: {v:4d}  ({v/len(msgs)*100:.1f}%)")
    print(f"\nSaved: {op}\nSaved: {cp}")

if __name__=="__main__": main()
