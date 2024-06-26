## **What's New: GemPy v3 Release!**

**Introducing GemPy Version 3: Future-Proofing GemPy and its Role in the Open-Source Geoscience Ecosystem**

We are excited to announce the release of GemPy v3, which has been meticulously reworked to enhance robustness, reliability, and ease of installation. With this release, we aim to strengthen GemPy’s position as one of the cornerstones in the open-source geoscience ecosystem. Our ongoing commitment is to develop and provide a reliable and essential tool that promotes transparency and openness in geological modeling and beyond - thereby empowering users to realize their projects and implement novel scientific approaches.

**1. Transition from GemPy v2 to v3: Significant Changes**

While the core essence of GemPy remains unchanged, the upgrade from v2 to v3 introduces substantial changes to workflow steps and function names, potentially affecting familiar processes. To ensure that our users retain full access to previous functionalities and can continue running their existing GemPy v2-based projects, we have transitioned v2 to a package named [gempy_legacy](https://github.com/gempy-project/gempy_legacy).

We have updated our documentation and tutorials to reflect the changes in GemPy v3 and are preparing additional support materials, including short videos that showcase key functionalities of the new version.

Although the core GemPy team will not develop new features for the legacy version, we are committed to maintaining it based on community requests.

**2. Refined API**

The GemPy API has been extensively reworked to increase robustness and readiness for production environments. We've redesigned the API and data classes to optimize utility, minimize code repetition, and boost performance.

**3. Transitioning from Theano/Aesara to NumPy & PyTorch**

Due to discontinued support and frequent dependency issues with Theano/Aesara, we have decisively moved away from using these frameworks in GemPy (this transition also means that PyMC is no longer supported). Instead, to ensure GemPy's future-proofness, v3 incorporates a flexible tensor library framework that currently utilizes **`numpy`** and has optional dependencies on **`PyTorch`**. Our restructured code makes it easy to integrate additional tensor libraries in the backend. Additionally, similar functionalities to those provided by PyMC will be part of the upcoming **`gempy_probability`** module, which will enhance GemPy’s capabilities in probabilistic modeling.

**4. General Refactoring for Greater Robustness**

We've undertaken a refactoring of GemPy to enhance its robustness and reliability. Key improvements include:

- **Enhanced State Management:** We’ve made improvements to ensure consistently valid states in GemPy. Before, invalid states could arise during data adjustments.
- **Optimized Dependency Management:** We've streamlined the handling of dependencies. Most dependencies are now optional, which simplifies setup and integration for users.
- **Modular Design:** We've divided GemPy into several distinct libraries. This modular approach allows users to leverage specific functionalities according to their needs:
    - **`gempy_engine`** [here](https://github.com/gempy-project/gempy_engine): Handles core computational algorithms.
    - **`gempy_viewer`** [here](https://github.com/gempy-project/gempy_viewer): Provides visualization capabilities.
    - **`gempy_plugins`** [here](https://github.com/gempy-project/gempy_plugins): Supports additional functionalities through plugins.
    - **`gempy_probability`**: Focuses on probabilistic modeling and uncertainty analysis (Coming soon).
    - **`gempy`**: Now leaner and primarily focused on documentation and managing the API.

**5. Octree Implementation for Efficient Model Computation**

We have introduced an octree-based approach in GemPy v3 to optimize model computation and iteration times.

**6. Dual Contouring for High-Quality Meshing**

To ensure compatibility with our new octree approach, we have implemented dual contouring in GemPy v3. This method guarantees high-quality mesh generation, enhancing both the precision and visual appeal of 3D geological models.

**7. Initial Steps in LiquidEarth Integration**

We have begun integrating GemPy with the [LiquidEarth app](https://www.terranigma-solutions.com/liquid-earth-one) to significantly extend GemPy's applicability in practical scenarios, enhancing workflow integration and data visualization capabilities. LiquidEarth, developed by the same main developers as GemPy, is a commercial, cloud-based software solution. It empowers experts to visualize, edit, and communicate geological data and models in intuitive 3D, facilitating real-time collaboration across multiple devices, independent of location.

This integration aims to leverage LiquidEarth's commercial platform, which utilizes emerging technologies to provide novel environments for working with 3D geoscience data and focuses on ease of use and on offering a low entry barrier. Simultaneously, GemPy will continue to serve as a completely free and open-source tool, offering a flexible, reliable, and transparent solution for 3D geological modeling. Together, this direct integration between GemPy and LiquidEarth is designed to maximize the strengths of both systems: enhancing the visual representation of 3D geoscience projects, connecting them to the broader open-source geoscience ecosystem, and enhancing the utility of GemPy for companies and industry applications.

1. **Other Changes, Prototypes, and Future Developments**

GemPy v3 introduces numerous other smaller changes and feature prototypes. These include optimizations to the nugget effect for improved model stability, the inclusion of external implicit functions for modeling features such as dykes, and a prototype to accommodate fault zone thickness. Additionally, improved caching mechanisms have been implemented to enhance performance. One of the most requested features, which we plan to address in the near future, is the implementation of finite faults. Stay tuned for future updates as we continue to expand the GemPy documentation, publish additional examples, and progress with developments.

---

**In Conclusion**

With the release of v3, we are reinforcing the role of GemPy as a key contributing piece within the larger open-source geoscience ecosystem. This update not only improves core functionalities but also introduces important integrations and enhancements that make it more robust and pave the way for future developments. Dedicated to advancing GemPy as a valuable tool for both academic research and industry applications, our team values your ongoing support and feedback. We invite you to explore the new features and join us in shaping the future of open-source geosciences. Looking forward to the results and innovative applications you achieve with this new version of GemPy!
