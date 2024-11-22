## Issue resolution
- Closes #<ISSUE_NUMBER_HERE>

Select one: This pull request is...
- [ ] a bug fix: increment the patch version
- [ ] a small improvement: increment the minor version
- [ ] an incompatible (non-backwards compatible) API change: increment the major version

## 1. Does this do what we want it to do?

Objectives:
- Objective 1
- Objective 2
- ...
- Objective n

Required:
- [ ] Product Management: I have confirmed with the stakeholders that the objectives above are correct and complete.
- [ ] Testing: I have considered likely and/or severe edge cases and have included them in testing.

## 2. Are the implementation details accurate & efficient?

Required:
- [ ] Logic: I have visually inspected the entire pull request myself.
- [ ] Logic: I have left GitHub comments highlighting important pieces of code logic. I have had these code blocks reviewed by at least one other team member.

If applicable:
- [ ] Dependencies: This pull request introduces a new dependency. I have discussed this requirement with at least one other team member. The dependency is noted in `zppy-interfaces/conda`, not just an `import` statement.

## 3. Is this well documented?

Required:
- [ ] Documentation: by looking at the docs, a new user could easily understand the functionality introduced by this pull request.

## 4. Is this code clean?

Required:
- [ ] Readability: The code is as simple as possible and well-commented, such that a new team member could understand what's happening.
- [ ] Pre-commit checks: All the pre-commits checks have passed.

If applicable:
- [ ] Software architecture: I have discussed relevant trade-offs in design decisions with at least one other team member. It is unlikely that this pull request will increase tech debt.