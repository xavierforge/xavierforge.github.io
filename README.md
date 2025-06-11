# How to use?

![圖片](https://github.com/user-attachments/assets/9f86a376-3582-4d13-a441-bd94e458d338)

## Quick start

- Follow the official guides for [installation](https://hexo.io/zh-tw/docs/index.html) & [basic configuration](https://hexo.io/zh-tw/docs/configuration.html).
- Download the [miccall theme](https://github.com/miccall/hexo-theme-Mic_Theme) to `themes` directory -> Change the downloaded directory name to `miccall` -> Set it up in `_config.yml`:

  ```YAML
  # Extensions
  ## Plugins: https://hexo.io/plugins/
  ## Themes: https://hexo.io/themes/
  theme: miccall
  ```

- 🤖 Basic workflow: Edit ✍️ -> Commit ✅ -> Push to GitHub 🌎 (check [Deploy to GitHub Pages](https://hexo.io/zh-tw/docs/github-pages.html))

## Basic setup (`themes/miccall/_config.yml`)

1. Favicon, background image & profile picture:

   Put them in `themes/miccall/source/img/` with the name of `logo_miccall`, `bg`, and `me`, respectively.

2. Intro

   <img width="1000" alt="圖片" src="https://github.com/user-attachments/assets/eb87bebf-f259-4fc2-861c-d7553ba3e54b">

3. Nav - This section is pretty self-explanatory.

   <img width="1000" alt="圖片" src="https://github.com/user-attachments/assets/5a283d16-70f2-4b61-a79b-e6ec3cf8c47d">

   ```YAML
   Nav:
     Home_name: HOME # The name of the home page
     is_use_categories : true
     categories_name: Categories
     is_use_archives: false
     archives_name: Archives

     # The social icons on the nav bar
     # for more icons, please see http://fontawesome.io/icons/#brand
     icon:
       github: https://github.com/xavierforge
       linkedin: https://www.linkedin.com/in/chih-ying-yen/

     # Customized page
     # The value of link is the relative path to the directory in the source folder
     # Page:
     #   link: path
     pages:
       About:
         link: "/about/"
       Portfolio:
         link: "/portfolio/"
   ```

4. Tummy band

   <img width="1000" alt="圖片" src="https://github.com/user-attachments/assets/523d1194-a171-4948-9c64-e149a1c1f6b7">

   ```YAML
   # The main page under the nav bar
   MainFirst:
     name: Yen Chih Ying   # title
     description: Lifelong # secondary title
     pic_url: /img/me.jpg  # profile picture
     goto_ulr: ""          # link to jump to
   ```

5. Portfolio

   <img width="300" alt="圖片" src="https://github.com/user-attachments/assets/bacd7584-fb03-4784-8e69-6ccf9a8281f8">

   ```YAML
   Gallery:
     title: My Creations 🤖
     description: Anything worth doing well is worth doing poorly at first. - Brian Tracy<br>
   ```

## Usage

- `About` page

  1. Create an `about` directory in the `source` directory (This name corresponds to `like: "/about/"`).
  2. Create an `index.md` inside this folder, the content of about goes here.

- `Gallery` (`Portfolio`) page

  1. Create an `portfolio` directory in the `source` directory (This name corresponds to `like: "/portfolio/"`).
  2. Create an `index.md` inside this folder, and put the following attribute in [Front-matter](https://hexo.io/zh-tw/docs/front-matter):

     ```Markdown
     ---
     title: Portfolio
     date: 2023-05-02 21:39:03
     layout: gallery
     ---
     ```

  3. Create `gallery.yml` in `source/_data`, and put the projects in the following format:

     ```YAML
     project1:
       full_link: <URL for the main photo>
       thumb_link: <URL for the thumbnail>
       descr: <Description>
       link: <Link to Github>
     project2:
       full_link: <URL for the main photo>
       thumb_link: <URL for the thumbnail>
       descr: <Description>
       link: <Link to Github>
     ...
     ```

- Articles

  1. The folder fo articles is `source/_post`.
  2. Create a `.md` file with following structure (Use [`hexo new`](https://hexo.io/zh-tw/docs/commands#new)):

     ```Markdown
     ---
     title: # The title of ariticle
     date: 2017/3/27 13:48:25  # published time
     tags:
       - tag1
       - tag2 (Optional)
     categories: Algorithm
     thumbnail: https://xxxxxxxxxx.png
     ---

     CONTENT

     ```
