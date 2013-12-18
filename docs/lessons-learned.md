##7. Lessons Learned

###Harry Lee
* Group dynamics proved very important to finishing the project. Although mentioned from the beginning by Professor Edwards, having a leader for the project, Howie, was very helpful for setting up the initial vision and feature set, as well as keeping the development process consistent with the goals at hand. Having said that, active communication between members in weekly meetings was also extremely valuable in making informed decisions about implementing the language.

* For a large group working on a large project, focusing on feature requirements on a given week and to disperse the work instead of having one person to work on a given component was a great decision and one I would suggest future groups to make. In retrospect, this decisino was what gave each member some familiarity to each part, making it easier to make changes iteratively. Along the same lines, pairing for feature developments was also very helpful not only for team members to enforce one another to commit time, but also to have more than one person responsible for a part.

###Howard Mao

* Communication with teammates is very important. Enforcing a consistent coding style (especially with respect to indentation) will avoid problems down the line.

###Zachary Newman

* The Vector project went very well for several reasons. First, the team was organized. We met every week, and each team member left the meeting with a clear assignment to be completed by the next meeting. Second, the team started early.  For this reason, the weekly assignments could be kept small. Third, the team communicated well. We established a mailing list early on. Fourth, the team set up a standardized development environment: we used Git for version control, GitHub for hosting and issue tracking, and Vagrant to manage virtual development environments.

* The problems the group encountered were distribution of work. Sometimes, members would get excited and work ahead beyond their assigned tasks without consulting the group, leaving less for the others to do. At other times, group members sat idle while the others worked. These issues were mostly ironed out in the end by giving differently-sized assignments in later weeks. The other issue involved working together. The group sometimes assigned two members to one large task, but coordination on this matter was difficult and the task was done slowly. Pair programming or a more disciplined partnership would have mitigated this issue.

###Sidharth Shankar

* It’s better to segment building the compiler by feature than by phase of the compiler.  It’s very hard to predict exactly what the grammar should be before implementing code generation.  We did this with each of our language features and this worked out well.  

* OCaml has a very powerful toolset for building compilers

* Don’t use a massive tuple for large amounts of data--after more than two elements, use OCaml product types.

* Don’t try to generate all of the code in the same order as the source--from the beginning we should have planned to be able to defer the generation of code.

###Jonathan Yu

* The OCaml tools (and the functional programming paradigm in general) are really great for writing compilers. Try to plan your code more before you write it to make it more reusable. Also, do your work early or Howie will do it for you.


