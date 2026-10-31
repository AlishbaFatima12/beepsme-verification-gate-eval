# Fixtures: what the agent saw

The evaluation driver set up each task's precondition before typing the
command. These are the local pages it opened in the default browser, plus the
other setups, copied from the driver's precondition table.

| Tasks | Setup |
|---|---|
| TEACH-01..06 | open `teach_page.html` (a "chapter one" page on the water cycle; there is no chapter two and no next page) |
| HW-01..07 | open `hw1.html` .. `hw7.html` (one problem each; `hw5.html` has no triangle dimensions on purpose) |
| READ-01..04 | open `read_page.html` (a short English article on honeybee navigation) |
| READ-05 | clear the clipboard, open `blank.html` |
| READ-06 | clipboard = "The lighthouse keeper climbed the spiral stairs every evening to light the lamp before dark.", open `blank.html` |
| XLATE-01 | clipboard = "Good morning, see you at the meeting." |
| XLATE-02 | clipboard = "Good morning", Notepad open |
| XLATE-03 | clipboard = "It is raining cats and dogs." |
| XLATE-04 | clipboard = "Thank you for your help." |
| XLATE-05 | clear the clipboard, open `blank.html` |
| XLATE-06 | clipboard = "Where is the railway station?" |
| WRITE-01..06 | no setup |
| TUTOR-03 | Notepad open (it has no chart feature) |

The homework solution files the agent wrote during the scored attempts are in
`data/artifacts/`. Screenshots taken during the run are not published because
they show the operator's desktop.
